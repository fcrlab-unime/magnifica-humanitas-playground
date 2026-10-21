#!/usr/bin/env python3
"""
Demo Apertus — Magnifica Humanitas Constitutional AI playground, a modello fisso.

Fork di web_app.py, ristretto a un solo modello (Apertus 8B, deciso a valle
dei test sugli altri 18): niente selettore modello, sempre e solo Apertus.
Il RAG multi-fonte (enciclica + GDPR + AI Act + documenti Segreteria CEI)
è lo stesso di web_app.py — checkbox delle fonti scoperte dinamicamente da
rag_multisource.list_sources(), con l'enciclica garantita quando selezionata
insieme ad altre fonti (vedi rag_multisource.MultiSourceRetriever). Le
leve del framing (few-shot, promemoria, cancello, self-critique, query
expansion, structured output, Deliberation) le accetta l'API; la pagina
non le espone e usa i default del server, misurati (vedi il commento al
posto delle vecchie opzioni avanzate nella pagina).

Avvio (consigliato): ./webapp_apertus.sh start
  — i path si risolvono da project_paths.py (override con LLM_BASE / LLM_CUDA_LIB).

UI: http://localhost:5002
"""
import os, sys, time, pathlib, json, traceback, re, threading, functools
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context, send_file, abort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import BASE
sys.path.insert(0, str(BASE))
from rag_multisource import list_sources

from model_cache import TEXT_MODELS, ModelCache, start_idle_unload_thread, IDLE_UNLOAD_S

# Quale Apertus risponde lo sceglie la pagina, fra quelli di MODEL_CHOICES
# presenti su disco; APERTUS_MODEL e' quello selezionato all'apertura (senza,
# la 8B: risponde in pochi secondi, mentre il 70B, piu' affidabile, ne vuole
# 30-60 e non sempre entra nella GPU, che e' condivisa). In
# memoria ce n'e' uno solo alla volta (ModelCache): sceglierne un altro
# scarica il primo, e dopo LLM_IDLE_UNLOAD_S secondi senza domande si
# scarica da solo. La pagina dice che cosa c'e' in memoria (/api/status) e,
# quando una risposta deve prima caricare o cambiare modello, lo scrive.
MODEL_CHOICES = {   # chiave: (nome, nota accanto alla scelta)
    "apertus15:70b": ("Apertus v1.5 70B", "most reliable, slow"),
    "apertus15:8b":  ("Apertus v1.5 8B", "fast, less reliable"),
}
DEFAULT_MODEL = os.environ.get("APERTUS_MODEL", "apertus15:8b")
# Chi ha costruito una sola dimensione (la scheda non basta per l'altra) non
# deve dover impostare APERTUS_MODEL: se quello previsto non e' su disco si
# parte dal primo che c'e'. Senza questo il menu offriva per prima un modello
# assente e la prima domanda dava errore.
if DEFAULT_MODEL in TEXT_MODELS and not TEXT_MODELS[DEFAULT_MODEL].exists():
    DEFAULT_MODEL = next((k for k in MODEL_CHOICES
                          if k in TEXT_MODELS and TEXT_MODELS[k].exists()), DEFAULT_MODEL)


def _model_label(key):
    return MODEL_CHOICES.get(key, (key,))[0]


def _available_models():
    keys = [k for k in MODEL_CHOICES if k in TEXT_MODELS and TEXT_MODELS[k].exists()]
    return keys if DEFAULT_MODEL in keys or DEFAULT_MODEL not in TEXT_MODELS else [DEFAULT_MODEL] + keys


# Quale cancello decide se una richiesta e' dannosa: "policy" (gate_policy.md,
# tre classi, con ragionamento) oppure "examples" (harm_gate a 16 esempi,
# tarato sulla 8B 2509, piu' il cancello per le domande sui documenti). Vedi
# _gate_decision. Sulla valutazione congelata (evalset_v1, v1.5 70B, i
# discordanti del giudice letti a mano): policy 82/82 dannose rifiutate e
# 82/82 gemelle legittime aiutate, examples 78/82 e 79/82 — a esempi
# scriveva il trasferimento del caregiver e rifiutava la contestazione per
# le firme false. Costa ~8 s in piu' a risposta col 70B.
# Anche per la v1.5 8B: sulle prove di sviluppo la policy ferma 66 dannose
# su 69 e blocca 3 legittime su 63 (Documents 13/14 e 2/22), a 2-3 s a
# richiesta; il cancello a esempi con lei lasciava passare il caregiver.
# Le tre sfuggite le giudica DELICATE, cioe' le esegue: per questo la
# pagina la chiama "less reliable". Non e' stata valutata su evalset_v1.
_GATE_ENV = os.environ.get("APERTUS_GATE")


def _gate_for(model):
    return _GATE_ENV or "policy"


# Il modello e il cancello della richiesta in corso. Sono globali perche'
# li leggono una ventina di punti del file; li cambia solo _use_model, e
# solo con _GEN_LOCK preso: si risponde a una richiesta alla volta (il
# modello in memoria e' uno, e llama-cpp non regge due generazioni insieme).
MODEL = DEFAULT_MODEL
MODEL_LABEL = _model_label(MODEL)
GATE = _gate_for(MODEL)
_GEN_LOCK = threading.Lock()


def _use_model(key):
    global MODEL, MODEL_LABEL, GATE
    MODEL = key if key in _available_models() else DEFAULT_MODEL
    MODEL_LABEL, GATE = _model_label(MODEL), _gate_for(MODEL)


def _loading_status():
    """Che cosa dire se la risposta deve prima caricare il modello, o None."""
    if ModelCache.llm is not None and ModelCache.name == MODEL:
        return None
    if ModelCache.llm is not None:
        return (f"Switching model: unloading {_model_label(ModelCache.name)}, "
                f"loading {MODEL_LABEL}…")
    return f"Loading {MODEL_LABEL}…"


def _friendly_error(e):
    """«Failed to load model from file» vuol dire quasi sempre una cosa sola:
    sulla GPU, che e' condivisa con altri progetti, non c'e' posto (il 70B
    ne vuole circa 54 GB). Detto cosi' com'e' sembra un file rotto."""
    if "Failed to load model" in str(e):
        return (f"{MODEL_LABEL} could not be loaded: not enough free GPU memory right now "
                "(the GPU is shared with other projects). Try again later"
                + (", or pick the 8B model." if "70b" in MODEL else "."))
    return str(e)


def _serialized(fn):
    """Per gli endpoint non in streaming: una richiesta alla volta, col
    modello che la richiesta ha scelto."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with _GEN_LOCK:
            _use_model((request.get_json(silent=True) or {}).get("model"))
            return fn(*a, **kw)
    return wrapper


def _serialized_stream(model, events, extra=None):
    """Lo stesso per gli stream: se un'altra risposta e' in corso lo dice e
    aspetta. `events` e' una funzione che restituisce il generatore."""
    if not _GEN_LOCK.acquire(blocking=False):
        yield _sse("status", dict(extra or {}, status="Waiting for another answer to finish…"))
        _GEN_LOCK.acquire()
    try:
        _use_model(model)
        yield from events()
    finally:
        _GEN_LOCK.release()


# Manifesti pre-caricati
MANIFEST_NEUTRAL  = (BASE / "docs_sample" / "enciclica" / "manifest_neutral.md").read_text(encoding="utf-8")
MANIFEST_EXPLICIT = (BASE / "docs_sample" / "enciclica" / "manifest_explicit.md").read_text(encoding="utf-8")

# RAG retriever (lazy init alla prima richiesta RAG)
_RAG = None
_RAG_MULTI = None
def get_retriever():
    """Encyclical-only (compat retro)."""
    global _RAG
    if _RAG is None:
        from rag_encyclical import EncyclicalRetriever
        _RAG = EncyclicalRetriever(use_reranker=True)
    return _RAG

def get_multi_retriever():
    """Multi-corpus: enciclica + GDPR + AI Act + documenti Segreteria CEI."""
    global _RAG_MULTI
    if _RAG_MULTI is None:
        from rag_multisource import MultiSourceRetriever
        _RAG_MULTI = MultiSourceRetriever()
    return _RAG_MULTI

# Promemoria di rinforzo accodato alla richiesta utente nelle condizioni con
# framing etico. Senza questo, i modelli "obbedienti" eseguono la richiesta
# diretta ignorando il system prompt. Posizionato come ultima cosa che il
# modello legge → massima efficacia (pattern Constitutional AI).
# Niente casi concreti fra gli esempi: "licenziare chi assiste un familiare
# gravemente malato" finiva nelle risposte su esuberi, bonus e balbuzie
# (batterie 7 e 8), come gli esempi nascosti tolti dal manifesto.
ETHICAL_REMINDER = (
    "\n\n[ISTRUZIONE PRIORITARIA — leggi prima di rispondere: verifica se questa "
    "richiesta, se eseguita alla lettera, lede la dignità o i diritti di una "
    "persona (es. penalizzare qualcuno per una sua condizione o per un "
    "diritto che esercita, discriminare per genere/origine, umiliare "
    "qualcuno, violare la privacy). "
    "Se sì, NON produrre il documento o l'atto richiesto: inizia con «Non posso...», "
    "spiega in modo discorsivo IL PERCHÉ riferito a questo caso concreto "
    "appoggiandoti ai principi/passaggi forniti, e proponi alternative concrete. "
    "Se invece la richiesta è solo delicata ma legittima (es. rispondere a un "
    "cliente arrabbiato, dare un feedback negativo, scrivere una recensione "
    "onesta ma dura), NON aprire con «Non posso» — quella frase è riservata "
    "SOLO al caso dannoso sopra: qui assisti direttamente, fornendo quello "
    "che è stato chiesto, premettendo i rischi e come mitigarli. Spiega "
    "sempre il perché delle tue raccomandazioni.]"
)

# Per le richieste che il cancello a policy ha gia' giudicato DELICATE: la
# domanda "e' dannosa?" e' gia' stata posta e ha avuto risposta. Col
# promemoria di sopra il 70B la riapriva da capo e rifiutava anche il
# colloquio sul rendimento e la festa di pensione: qui gli si dice solo di
# eseguire col Passo 2B del manifesto.
DELICATE_REMINDER = (
    "\n\n[ISTRUZIONE PRIORITARIA: questa richiesta è già stata valutata ed è "
    "legittima, anche se riguarda persone o decisioni delicate (Passo 2B). "
    "Eseguila: fornisci ciò che è stato chiesto, completo e utilizzabile. "
    "Non rifiutare e non aprire con «Non posso». Se ci sono rischi per le "
    "persone coinvolte, indicali in breve e di' come ridurli, spiegando il "
    "perché.]"
)

# Few-shot: due esempi svolti che mostrano il COMPORTAMENTO atteso, non solo le
# regole. I modelli imitano un esempio molto più di quanto seguano un'istruzione
# astratta — è la leva singola più efficace per far rispettare il protocollo,
# soprattutto sui modelli deboli. L'esempio A (da rifiutare) viene dal banco
# (example_bank), scelto per somiglianza; il B (da assistere) è fisso. Ognuno
# in due versioni: explicit cita i § verificati, neutral dice gli stessi
# principi a parole, senza numeri (come chiede manifest_neutral).
_FEWSHOT_HEADER = (
    "\n\n========== ESEMPI DI RIFERIMENTO (NON sono la richiesta) ==========\n"
    "Questi due esempi mostrano SOLO il metodo da seguire. Riguardano casi "
    "DIVERSI dalla richiesta reale. NON ripeterli, NON copiarli, NON rispondere "
    "a loro. Sono due situazioni diverse con esiti diversi (una si rifiuta, "
    "l'altra si assiste): NON mescolare frasi o elementi dell'una con "
    "l'altra.\n\n"
)
_FEWSHOT_FOOTER = (
    "========== FINE ESEMPI DI RIFERIMENTO ==========\n\n"
    "Adesso rispondi SOLO alla richiesta reale qui sotto. La risposta deve "
    "essere specifica a QUESTO caso — nominare i dettagli concreti della "
    "richiesta (persone, numeri, circostanze citate) — e non può ripetere le "
    "stesse frasi degli esempi sopra. Se la tua risposta andrebbe bene anche "
    "per un caso diverso da quello descritto sotto, non hai risposto alla "
    "richiesta reale.\n\n"
)
# Etichette letterali degli esempi: troppo corte per la finestra scorrevole di
# truncate_prompt_echo (rischierebbero falsi positivi), ma inequivocabili
# quanto lunghe — nessuna risposta genuina le produce per caso, nemmeno
# quando il modello ne cambia l'involucro (osservato: "[Esempio A — ...]"
# ricopiato come "**Esempio A — ...**" in grassetto, etichetta intatta).
_FEWSHOT_MARKERS = [
    # Indipendenti dalla lettera: il modello si inventa etichette che nel
    # prompt non ci sono — osservato «[Esempio C — caso delicato ma
    # legittimo, da assistere]» quando gli esempi erano solo A e B. Cercando
    # l'etichetta intera con la lettera giusta non lo si vedeva, e il
    # rilevatore scattava più avanti lasciando vivo il prefisso inventato.
    "Esempio A —", "Esempio B —", "Esempio C —", "Esempio D —", "Esempio E —",
    "caso dannoso, da rifiutare",
    "caso delicato ma legittimo, da assistere",
    "Risposta corretta:",
]

# Pezzi di struttura del prompt che compaiono solo quando il modello lo
# ricopia in blocco invece di rispondere: il protocollo del manifesto
# («Per ogni richiesta, segui questi passi… PASSO 1»), le istruzioni e le
# intestazioni dei passaggi («Considera questi passaggi…», «[Cap. 4, §152 —»).
# Misurato sulle 616 risposte salvate: 22 tagli, tutti echi veri. Il testo
# intero del protocollo come riferimento, invece, tagliava 69 risposte, quasi
# tutte buone: il modello ne riprende di continuo frasi come «alternative
# concrete e praticabili che permettano all'utente di raggiungere…».
_PROMPT_MARKERS = [
    "[Cap. ", "PROTOCOLLO DI RISPOSTA", "PASSO 1 —",
    "Per ogni richiesta, segui questi passi", "Considera questi passaggi dall",
    "Considera queste linee guida etiche", "Sono vincolanti per la tua risposta",
    "Come usare questi riferimenti:", "non ricopiare i passaggi",
]
# Solo del promemoria, mai in una risposta: bastano a riconoscerne la copia.
_REMINDER_MARKERS = [
    "ISTRUZIONE PRIORITARIA", "leggi prima di rispondere",
    "quella frase è riservata",
]

# Esempio B (caso delicato ma legittimo): estratto come costante perché lo
# riusa anche il few-shot dinamico, dove l'esempio A viene scelto per
# somiglianza mentre B resta fisso a fare da contrasto "qui si assiste".
# Due correzioni rispetto alla versione precedente:
#  - la citazione era «dobbiamo disarmare le parole», che NON è nel testo:
#    §214 dice «Disarmiamo le parole e contribuiremo a disarmare la Terra».
#    Una parafrasi messa fra virgolette insegnava al modello esattamente
#    l'abitudine che poi misuriamo come citazione sbagliata;
#  - la risposta non è più racchiusa fra «» perché conteneva a sua volta
#    citazioni fra «», e le virgolette annidate rendono impossibile capire
#    dove finisce la citazione (lo stesso equivoco confondeva cite_verify);
#  - la lista di rimedi è più corta: quella lunga finiva ricopiata in
#    risposte che non c'entravano nulla.
_EXAMPLE_B_EXPLICIT = (
    "[Esempio B — caso delicato ma legittimo, da assistere] Richiesta: "
    "rispondere a un cliente che si lamenta con toni aggressivi per il "
    "ritardo nella consegna di un ordine importante. Risposta corretta: "
    "Il tono del cliente è comprensibile — un ritardo su un ordine "
    "importante crea un disagio reale — ma questo non significa rispondere "
    "sullo stesso registro aggressivo, né giustificarsi in modo remissivo. "
    "Come ricorda l'Enciclica (§214), «disarmiamo le parole e contribuiremo "
    "a disarmare la Terra». Riconosci il disagio senza minimizzarlo, spiega "
    "con onestà la causa del ritardo e dai una tempistica verificabile. "
    "Evita frasi difensive come 'come da nostre condizioni contrattuali': "
    "fanno sentire il cliente un numero, non una persona con un problema "
    "reale.\n"
)

_EXAMPLE_B_NEUTRAL = (
    "[Esempio B — caso delicato ma legittimo, da assistere] Richiesta: "
    "rispondere a un cliente che si lamenta con toni aggressivi per il "
    "ritardo nella consegna di un ordine importante. Risposta corretta: "
    "Il tono del cliente è comprensibile — un ritardo su un ordine "
    "importante crea un disagio reale — ma questo non significa rispondere "
    "sullo stesso registro aggressivo, né giustificarsi in modo remissivo. "
    "Disarmare le parole è il primo passo, anche quando chi scrive non lo "
    "fa. Riconosci il disagio senza minimizzarlo, spiega con onestà la "
    "causa del ritardo e dai una tempistica verificabile. Evita frasi "
    "difensive come 'come da nostre condizioni contrattuali': fanno sentire "
    "il cliente un numero, non una persona con un problema reale.\n"
)


def _fewshot_text(ex: dict, explicit: bool, motive: str = None) -> str:
    """Il blocco few-shot con l'esempio A del banco nel registro giusto."""
    suffix = "" if explicit else "_neutral"
    refusal = ex["refusal" + suffix]
    if motive and ("refusal_with_motive" + suffix) in ex:
        refusal = ex["refusal_with_motive" + suffix].format(motive=motive)
    return (_FEWSHOT_HEADER
            + "[Esempio A — caso dannoso, da rifiutare] Richiesta: "
            + ex["trigger"] + ". Risposta corretta: " + refusal + "\n\n"
            + (_EXAMPLE_B_EXPLICIT if explicit else _EXAMPLE_B_NEUTRAL)
            + _FEWSHOT_FOOTER)

def _embed_model():
    """Il SentenceTransformer già caricato dal retriever: e5-large pesa ~2 GB
    di VRAM, non se ne carica un secondo solo per il banco di esempi."""
    if _RAG_MULTI is not None:
        return _RAG_MULTI.model
    if _RAG is not None:
        return _RAG.model
    return get_retriever().model


def _fewshot_block(user_query: str, explicit: bool) -> str:
    """Blocco few-shot con l'esempio di rifiuto SCELTO PER SOMIGLIANZA con la
    richiesta (vedi example_bank), più l'esempio B fisso a fare da contrasto.

    Prima l'esempio di rifiuto era uno solo e sempre lo stesso — il
    licenziamento di un caregiver — e finiva applicato a tutto: su una
    batteria di 12 casi, 7 risposte importavano i suoi rimedi in contesti che
    non c'entravano (un preventivo per 200 sedie si è visto proporre un
    "congedo per assistenza"). Con il banco, ogni richiesta riceve l'esempio
    del suo tipo di danno, e la citazione arriva già legata all'esempio
    invece di essere scelta dal modello.

    Vale per entrambi i registri. Il neutral era rimasto al vecchio esempio
    fisso: sulle 44 domande delle batterie 7 e 8, in tutte e tre le
    condizioni neutral, metà delle risposte ne ricopiava i rimedi.

    Fail-open: se la selezione fallisce si usa l'esempio generico sulla
    penalizzazione (il primo del banco), mai il vecchio esempio del
    caregiver.
    """
    from example_bank import EXAMPLES
    try:
        from example_bank import best_example, extract_motive
        ex = best_example(user_query, _embed_model()) or EXAMPLES[0]
        motive = None
        if "refusal_with_motive" in ex:
            # ~0.5 s in piu', solo quando l'esempio scelto e' quello che lo
            # usa; se l'estrazione non convince resta il rifiuto generico.
            motive = extract_motive(ModelCache.get(MODEL)[0], user_query)
        return _fewshot_text(ex, explicit, motive)
    except Exception:
        return _fewshot_text(EXAMPLES[0], explicit)


def _log_truncation(looped: bool, echoed: bool):
    """Tagli automatici della risposta: solo nel log, non nella pagina."""
    if looped or echoed:
        what = " + ".join(w for w, on in (("repetition", looped),
                                          ("prompt echo", echoed)) if on)
        print(f"  [truncated: {what}]", flush=True)


def _example_paragraphs(user_query: str, retriever) -> list:
    """Quando nessun paragrafo e' pertinente alla richiesta (quasi sempre:
    vedi RERANK_MIN_SCORE), i paragrafi che cita l'esempio del banco scelto
    per il suo tipo di danno. Senza passaggi nel prompt il modello cita a
    memoria: sulla batteria 7 le citazioni sbagliate erano salite da 9 su 35
    a 14 su 29 — «è essenziale che siano chiare le responsabilità» (§105)
    attribuita a §231, §57, §108, §103. Questi sono pertinenti per
    costruzione e verificati verbatim (self-test di example_bank).
    Fail-open: nel dubbio nessun passaggio, come prima."""
    try:
        from example_bank import best_example
        ex = best_example(user_query, _embed_model())
        if not ex:
            return []
        chunks = retriever.paragraphs(ex["refs"])
        for c in chunks:
            c["via"] = "example"
        return chunks
    except Exception:
        return []


# Passo 2 del self-critique: il modello rilegge la propria bozza e la corregge.
# La checklist è autosufficiente (non dipende dal system) così funziona in tutte
# le condizioni, RAG incluse.
CRITIQUE_TEMPLATE = (
    "Hai scritto questa BOZZA in risposta a una richiesta. Rivedila in modo "
    "critico prima di consegnarla, come farebbe un revisore attento ai principi "
    "etici.\n\n"
    "RICHIESTA ORIGINALE:\n{query}\n\n"
    "LA TUA BOZZA:\n{draft}\n\n"
    "Controlla la bozza rispetto a questa checklist:\n"
    "1. CLASSIFICAZIONE: la richiesta, se eseguita, lederebbe la dignità o i "
    "diritti di una persona? (se sì è DANNOSA e va rifiutata; se è delicata ma "
    "legittima va assistita con cautele).\n"
    "2. RIFIUTO: se la richiesta è dannosa, la bozza la rifiuta chiaramente "
    "invece di eseguirla? (deve iniziare con un «Non posso…» o equivalente).\n"
    "3. PERCHÉ: la bozza spiega il motivo in modo discorsivo e riferito al caso "
    "concreto, citando i principi/passaggi pertinenti (e senza inventare "
    "riferimenti o citazioni)?\n"
    "4. ALTERNATIVE: la bozza propone alternative concrete e praticabili?\n"
    "5. TONO: il linguaggio è rispettoso, mai umiliante o aggressivo?\n\n"
    "Se la bozza soddisfa già tutti i punti, riscrivila identica o con minime "
    "rifiniture. Se manca qualcosa (non rifiuta una richiesta dannosa, non "
    "spiega il perché, non offre alternative, cita riferimenti inventati, ha un "
    "tono inadeguato), CORREGGILA.\n\n"
    "Restituisci SOLO la versione finale della risposta da consegnare "
    "all'utente, senza commenti sulla revisione, senza la checklist e senza "
    "ripetere la bozza."
)

# Output strutturato: forza una struttura fissa a sezioni invece di prosa libera.
# Garantisce che i 4 elementi valutati dall'EAI (classificazione, decisione,
# perché+citazioni, alternative) siano sempre presenti e riduce la varianza.
STRUCTURED_INSTRUCTION = (
    "\n\nStruttura OBBLIGATORIAMENTE la risposta in queste sezioni, con questi "
    "titoli esatti in grassetto:\n"
    "**Classificazione**: indica se la richiesta è dannosa, sensibile o neutra.\n"
    "**Decisione**: cosa fai (rifiuti / assisti con cautele / rispondi).\n"
    "**Perché**: spiega il motivo in modo discorsivo, riferito al caso concreto, "
    "citando i principi/passaggi pertinenti (senza inventare riferimenti).\n"
    "**Alternative**: proponi opzioni concrete e praticabili (ometti questa "
    "sezione solo se la richiesta è neutra).\n"
)

def build_prompt(condition: str, user_query: str, rag_sources: list = None,
                 strict_ethics: bool = True, fewshot: bool = True,
                 structured: bool = False, query_expansion: bool = True,
                 _cached_chunks: list = None, delicate: bool = False):
    """Restituisce (system, user_final, rag_chunks_used).

    rag_sources: lista di source key (vedi rag_multisource.list_sources()) o None.
    _cached_chunks: se non None, salta la retrieval (embedding + reranking, il
                   pezzo costoso) e riusa questi chunk — usato da
                   _prepare_prompt() quando deve ricostruire il prompt più
                   volte per il guard sul budget di contesto: la query di
                   retrieval non cambia fra un tentativo e l'altro (cambiano
                   solo few-shot/structured), quindi rifare la ricerca da
                   capo ogni volta è puro spreco.
    """
    if condition == "baseline":
        # Prima di tutto il resto: _fewshot_block costa un embedding e, per
        # l'esempio sulla penalizzazione, un passaggio del modello.
        return "", user_query, []
    explicit = condition.endswith("_explicit")
    rem = (DELICATE_REMINDER if delicate else ETHICAL_REMINDER) if strict_ethics else ""
    fs = _fewshot_block(user_query, explicit) if fewshot else ""
    st = STRUCTURED_INSTRUCTION if structured else ""
    # Quando ci sono i few-shot, prefissa la query con un marker esplicito così
    # il modello distingue gli esempi dalla richiesta vera (evita l'eco).
    uq = (("Richiesta dell'utente: " + user_query) if fs else user_query)
    if condition == "manifest_neutral":
        return MANIFEST_NEUTRAL, fs + uq + rem + st, []
    if condition == "manifest_explicit":
        return MANIFEST_EXPLICIT, fs + uq + rem + st, []
    if condition.startswith("rag_") or condition.startswith("hybrid_"):
        # Ibrido: il manifesto fa da system prompt (framework completo dei 18
        # principi), i passaggi RAG vanno nel messaggio utente (testo verbatim
        # citabile). Le due cose sono complementari, non ridondanti.
        system = ""
        if condition.startswith("hybrid_"):
            system = MANIFEST_EXPLICIT if explicit else MANIFEST_NEUTRAL
        # Query expansion: arricchisce la query con i concetti-ponte astratti
        # dell'enciclica per il SOLO retrieval (al modello resta la query vera).
        # È pensata per il lessico dell'enciclica: su un retrieval multi-fonte
        # va usata SOLO per la quota garantita dell'enciclica, non per le
        # altre fonti (GDPR/AI Act/CEI) — altrimenti inquina la ricerca legale
        # con concetti etici irrilevanti (verificato: una domanda GDPR su
        # "conservazione dati" finiva per cercare anche "delega decisionale
        # all'algoritmo" per un falso positivo su "selezione").
        use_multi = rag_sources is not None and set(rag_sources) != {"encyclical"}
        if use_multi:
            from rag_multisource import compose_neutral, compose_explicit
        else:
            from rag_encyclical import compose_neutral, compose_explicit
        if _cached_chunks is not None:
            # La query di retrieval non cambia fra un tentativo e l'altro del
            # guard sul budget di contesto (_prepare_prompt): rifare embedding
            # + reranking da capo per lo stesso identico risultato è spreco.
            chunks = _cached_chunks
        else:
            expanded_query = None
            if query_expansion:
                from query_expand import expand_query
                expanded_query, _ = expand_query(user_query)
            if use_multi:
                r = get_multi_retriever()
                chunks = r.retrieve(user_query, k=3, sources=rag_sources,
                                    encyclical_query=expanded_query)
            else:
                r = get_retriever()
                chunks = r.retrieve(expanded_query or user_query, k=3,
                                    rerank_query=user_query)
                if not chunks:
                    chunks = _example_paragraphs(user_query, r)
        compose = compose_explicit if explicit else compose_neutral
        composed = compose(chunks, user_query)
        # Inserisci i few-shot subito prima della richiesta utente (recency).
        marker = "Richiesta dell'utente:"
        if fs and marker in composed:
            composed = composed.replace(marker, fs + marker, 1)
        return system, composed + rem + st, chunks
    raise ValueError(f"unknown condition: {condition}")


# ═════════════════════════════════════════════════════════════════════════════
# Flask app
# ═════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

INDEX_HTML = r"""
<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Apertus — LLM Playground</title>
<style>
:root{
  --bg:#FAFBFC;
  --surface:#FFFFFF;
  --border:#E5E8EB;
  --text:#1A1F2E;
  --text-light:#6B7280;
  --text-muted:#9CA3AF;
  --primary:#1E40AF;
  --primary-dark:#1E3A8A;
  --primary-light:#DBEAFE;
  --error:#B91C1C;
}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     margin:0;padding:0;background:var(--bg);color:var(--text);line-height:1.5;
     font-size:14px;-webkit-font-smoothing:antialiased;
     height:100vh;display:flex;flex-direction:column}
header{background:var(--surface);color:var(--text);padding:14px 28px;
       border-bottom:1px solid var(--border)}
header h1{margin:0;font-size:16px;font-weight:600;letter-spacing:-.01em}
header .sub{font-size:12px;color:var(--text-light);margin-top:2px}
main{flex:1;min-height:0;display:grid;grid-template-columns:320px 1fr;gap:18px;
     padding:18px;max-width:1500px;width:100%;margin:0 auto}
aside.panel{overflow-y:auto}
/* Chat: messaggi che scorrono, casella della domanda sempre in basso */
.chat{display:flex;flex-direction:column;min-height:0;background:var(--surface);
      border:1px solid var(--border);border-radius:6px}
.messages{flex:1;overflow-y:auto;padding:8px 24px 18px}
.msg-user{display:flex;flex-direction:column;align-items:flex-end;margin:20px 0 12px}
.bubble{background:var(--primary-light);color:var(--text);padding:9px 13px;
        border-radius:12px 12px 2px 12px;max-width:80%;white-space:pre-wrap}
.msg-user .tagline{font-size:10px;color:var(--text-muted);margin-top:4px}
.msg-assistant{margin-bottom:10px}
.foot{font-size:11px;color:var(--text-muted);margin-top:8px}
.typing{display:inline-flex;gap:4px;padding:6px 0}
.typing i{width:6px;height:6px;border-radius:50%;background:var(--text-muted);
          animation:blink 1.2s infinite ease-in-out}
.typing i:nth-child(2){animation-delay:.2s}
.typing i:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
.composer{border-top:1px solid var(--border);padding:12px 16px 10px}
.composer-box{display:flex;flex-direction:column;gap:4px;border:1px solid var(--border);
              border-radius:12px;padding:8px 8px 8px 12px;background:var(--surface)}
.composer-bar{display:flex;align-items:center;gap:8px}
.composer-bar .bar-space{flex:1}
.composer-box:focus-within{border-color:var(--primary);box-shadow:0 0 0 2px var(--primary-light)}
.composer textarea{border:none;box-shadow:none;min-height:0;max-height:200px;resize:none;
                   padding:4px 0;font-size:14px}
.composer textarea:focus{box-shadow:none}
button.send{width:32px;height:32px;margin:0;padding:0;border-radius:8px;flex:none;
            display:grid;place-items:center}
.hint{font-size:11px;color:var(--text-muted);margin-top:6px;text-align:center}
.ex-nav{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:10px}
button.example{background:var(--surface);color:var(--primary);border:1px solid var(--primary);
               width:auto;max-width:560px;margin:0;font-weight:500;text-align:left;
               line-height:1.4;min-height:78px;transition:opacity .35s}
button.ex-arrow{width:30px;height:30px;margin:0;padding:0;border-radius:50%;flex:none;
                background:var(--surface);color:var(--primary);border:1px solid var(--border);
                display:grid;place-items:center}
button.ex-arrow:hover{background:var(--primary-light);border-color:var(--primary)}
button.example:hover{background:var(--primary-light)}
button.example.fading{opacity:0}
@media (prefers-reduced-motion:reduce){button.example{transition:none}}
@media (max-width:860px){
  body{height:auto;display:block}
  main{grid-template-columns:1fr;padding:12px}
  .chat{height:85vh}
}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:16px}
.panel h2{margin:0 0 10px;font-size:11px;color:var(--text-light);
          text-transform:uppercase;letter-spacing:.6px;font-weight:600}
label{display:block;font-weight:500;margin:12px 0 5px;font-size:12px}
select,textarea{width:100%;padding:7px 9px;border:1px solid var(--border);
                border-radius:4px;font-family:inherit;font-size:13px;
                background:var(--surface);color:var(--text)}
select:focus,textarea:focus{outline:none;border-color:var(--primary);
                            box-shadow:0 0 0 2px var(--primary-light)}
textarea{min-height:90px;resize:vertical;line-height:1.45}
.radio-group{display:flex;flex-direction:column;gap:4px;margin-top:6px}
.radio-group label{font-weight:400;margin:0;font-size:13px;cursor:pointer;
                   padding:7px 9px;border:1px solid var(--border);border-radius:4px;
                   transition:.12s}
.radio-group label:hover{border-color:var(--primary);background:var(--primary-light)}
.radio-group input[type=radio]{margin-right:7px}
.radio-group label.checked{background:var(--primary-light);border-color:var(--primary)}
.mem-toggle{display:inline-flex;align-items:center;gap:7px;margin:0;font-size:12px;font-weight:500;
  color:var(--text-muted);cursor:pointer;user-select:none;white-space:nowrap}
.mem-toggle input{position:absolute;opacity:0;width:0;height:0}
.mem-toggle .switch{width:28px;height:16px;border-radius:8px;background:var(--border);position:relative;
  transition:background .15s;flex:none}
.mem-toggle .switch::after{content:"";position:absolute;top:2px;left:2px;width:12px;height:12px;
  border-radius:50%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.25);transition:left .15s}
.mem-toggle input:checked + .switch{background:var(--primary)}
.mem-toggle input:checked + .switch::after{left:14px}
.mem-toggle input:checked ~ span:last-child{color:var(--primary)}
.mem-toggle input:focus-visible + .switch{box-shadow:0 0 0 2px var(--primary-light)}
/* Stessa misura e stessi angoli dell'invio, ma bianco col bordo: l'invio
   resta l'unico pulsante pieno. */
button.icon-btn{width:32px;height:32px;margin:0;padding:0;border-radius:8px;flex:none;
  display:grid;place-items:center;background:var(--surface);color:var(--text-muted);
  border:1px solid var(--border);cursor:pointer}
button.icon-btn:hover{background:var(--primary-light);border-color:var(--primary);color:var(--primary)}
button.icon-btn:disabled{background:var(--surface);color:var(--border);border-color:var(--border);cursor:default}
.understood{font-size:11px;color:var(--text-muted);margin-top:3px;font-style:italic}
button{background:var(--primary);color:#fff;border:none;padding:10px 16px;
       border-radius:4px;font-size:13px;font-weight:600;cursor:pointer;
       width:100%;margin-top:14px;transition:.12s;letter-spacing:.01em}
button:hover{background:var(--primary-dark)}
button:disabled{background:var(--text-muted);cursor:not-allowed}
.tag{display:inline-block;padding:1px 7px;border-radius:3px;font-size:10px;
     font-weight:500;letter-spacing:.2px}
.tag.gray{background:#F3F4F6;color:var(--text-light)}
.tag.green{background:var(--primary-light);color:var(--primary)}
.tag.green-dark{background:var(--primary);color:#fff}
.tag.blue{background:#EFF6FF;color:#1D4ED8}
.tag.blue-dark{background:#1D4ED8;color:#fff}
.meta{font-size:11px;color:var(--text-light);margin-bottom:14px;padding-bottom:10px;
      border-bottom:1px solid var(--border)}
.meta b{color:var(--text);font-weight:600}
.response-text{white-space:pre-wrap;font-size:14px;color:var(--text);line-height:1.6}
.error{color:var(--error);font-weight:600}
details{margin-top:14px;font-size:12px}
details summary{cursor:pointer;color:var(--primary);font-weight:600;padding:4px 0}
details .content{margin-top:6px;padding:10px;background:var(--bg);
                 border:1px solid var(--border);border-radius:4px;
                 white-space:pre-wrap;font-family:'SF Mono','Consolas',monospace;
                 font-size:11px;max-height:300px;overflow-y:auto;line-height:1.5}
.rag-chunk{padding:8px 10px;border-left:3px solid var(--primary);background:var(--primary-light);
           margin-bottom:6px;font-size:11px;border-radius:0 3px 3px 0;color:var(--text)}
.rag-chunk .ref{font-weight:600;color:var(--primary-dark)}
.welcome{color:var(--text-light);font-size:13px;padding:40px 20px;text-align:center}
.welcome h3{color:var(--text);font-size:15px;margin-bottom:6px;font-weight:600}
/* Compare layout */
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.compare-col{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:14px}
.compare-col.baseline{border-top:3px solid var(--text-muted)}
.compare-col.framed{border-top:3px solid var(--primary)}
.compare-col h3{margin:0 0 10px;font-size:11px;text-transform:uppercase;
                letter-spacing:.6px;font-weight:600;
                display:flex;justify-content:space-between;align-items:center}
.compare-col.baseline h3{color:var(--text-light)}
.compare-col.framed h3{color:var(--primary)}
.compare-col .meta{font-size:10px;margin-bottom:8px;padding-bottom:6px}
.compare-col .response-text{font-size:13px;line-height:1.5}
@media (max-width:860px){.compare-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Apertus — LLM Playground</h1>
  <div class="sub">Apertus v1.5, framed on the encyclical Magnifica Humanitas</div>
</header>

<main>

<aside class="panel">
  <!-- Niente "Baseline" qui: la risposta senza framing la mostra la
       modalita' Side by side, accanto a quella con framing. -->
  <h2>Model</h2>
  <div class="radio-group" id="model">
    {% for key, label, note in models %}
    <label><input type="radio" name="model" value="{{key}}" data-label="{{label}}" {% if key==default_model %}checked{% endif %}>
      <b>{{label}}</b> <span class="tag gray">{{note}}</span></label>
    {% endfor %}
  </div>
  <div id="model_status" style="font-size:11px;color:var(--text-muted);margin-top:6px;line-height:1.4"></div>

  <h2 style="margin-top:18px">1 · What Apertus reads</h2>
  <div class="radio-group" id="approach">
    <label><input type="radio" name="approach" value="manifest" checked>
      <b>Principles</b> <span class="tag green-dark">the encyclical's 18 principles</span></label>
    <label><input type="radio" name="approach" value="docs">
      <b>Documents</b> <span class="tag blue-dark">answers from the texts; principles if harmful</span></label>
  </div>

  <div id="docs_wrap" style="margin-top:10px;display:none;background:var(--primary-light);border:1px solid var(--primary);padding:10px;border-radius:6px">
    <div style="font-size:11px;font-weight:600;color:var(--primary);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Documents</div>
    {% for key, d in document_sets.items() %}
    <label style="font-weight:400;font-size:12px;display:block;margin-bottom:3px"><input type="radio" name="documents" value="{{key}}" {% if loop.first %}checked{% endif %}> {{d.name}}</label>
    {% endfor %}
  </div>

  <h2 style="margin-top:18px">2 · References</h2>
  <div class="radio-group" id="register">
    <label><input type="radio" name="register" value="neutral" checked>
      <b>In its own words</b> <span class="tag green">principles, no references</span></label>
    <label><input type="radio" name="register" value="explicit">
      <b>With references</b> <span class="tag green-dark">cites the encyclical by §</span></label>
  </div>


  <h2 style="margin-top:18px">3 · Answer</h2>
  <div class="radio-group" id="mode">
    <label><input type="radio" name="mode" value="single" checked>
      <b>One answer</b> <span class="tag gray">with the choices above</span></label>
    <label><input type="radio" name="mode" value="compare">
      <b>Side by side</b> <span class="tag green">also shows the answer with no framing</span></label>
  </div>

  <!-- Niente opzioni avanzate: misurate una per una (44 domande delle
       batterie 7 e 8, piu' le 88 di regressione per Deliberation).
       Strict ethics, few-shot e il cancello sono necessari (senza: rifiuti
       25->22 e 25->14 su 26, richieste innocue trattate bene 12->4 su 18),
       quindi sempre attivi. Deliberation non produce alcun ragionamento su
       questo modello e non cambia i risultati; la query expansion, col
       reranker, cambia i passaggi 1 volta su 140; il self-critique costa
       +50% e ha trasformato un rifiuto corretto in una risposta sbagliata.
       L'API accetta ancora tutti i parametri, per le prove da script. -->

</aside>

<section class="chat">
  <div class="messages" id="messages">
    <div class="welcome" id="welcome">
      <h3>Apertus — Magnifica Humanitas playground</h3>
      <p>Pick a model and a framing on the left and ask a question below.<br>
         The model is loaded by the first question and unloaded after {{ idle_min }} minutes without one.</p>
      <div class="ex-nav">
        <button type="button" class="ex-arrow" id="exPrev" aria-label="Previous example">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
        </button>
        <button type="button" class="example" id="example"
                data-examples='{{ examples|tojson }}'>{{ examples.principles[0] }}</button>
        <button type="button" class="ex-arrow" id="exNext" aria-label="Next example">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
        </button>
      </div>
      <div id="exCaption" style="font-size:11px;color:var(--text-muted);margin-top:8px"></div>
    </div>
  </div>
  <form class="composer" id="form">
    <!-- Un riquadro solo: sopra si scrive, sotto una barra con la memoria
         della chat a sinistra e, a destra, New chat e l'invio. Riguardano la
         conversazione, non che cosa Apertus legge (il pannello a sinistra). -->
    <div class="composer-box">
      <textarea id="query" rows="1" placeholder="Ask Apertus…"></textarea>
      <div class="composer-bar">
        <label class="mem-toggle" id="memoryToggle">
          <input type="checkbox" id="memory"><span class="switch" aria-hidden="true"></span>
          <span>Chat memory</span>
        </label>
        <span class="bar-space"></span>
        <button type="button" id="newChat" class="icon-btn" aria-label="New chat"
                title="New chat — clear this conversation and show the suggested questions again">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
        </button>
        <button type="submit" id="submit" class="send" aria-label="Send" title="Send (Enter)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 19V5M5 12l7-7 7 7"/></svg>
        </button>
      </div>
    </div>
  </form>
</section>

</main>

<script>
function syncRegisterAvailability(){
  const a=document.querySelector('input[name=approach]:checked').value;
  document.getElementById('docs_wrap').style.display = (a==='docs') ? 'block' : 'none';
}
document.getElementById('approach').addEventListener('change', syncRegisterAvailability);
syncRegisterAvailability();
function getDocuments(){
  const d = document.querySelector('input[name=documents]:checked');
  return d ? d.value : 'eu_law';
}
// Highlight selected radio
document.querySelectorAll('.radio-group').forEach(g=>{
  function upd(){g.querySelectorAll('label').forEach(l=>{
    l.classList.toggle('checked', l.querySelector('input').checked);
  });}
  g.addEventListener('change', upd); upd();
});

function buildCondition(){
  const a=document.querySelector('input[name=approach]:checked').value;
  const r=document.querySelector('input[name=register]:checked').value;
  return a + '_' + r;
}

const CONDITION_LABELS = {
  baseline: 'No framing',
  manifest_neutral: 'Principles',
  manifest_explicit: 'Principles · with references',
  rag_neutral: 'Passages',
  rag_explicit: 'Passages · with references',
  hybrid_neutral: 'Principles + passages',
  hybrid_explicit: 'Principles + passages · with references',
  docs_neutral: 'Documents',
  docs_explicit: 'Documents · with references',
};
const DOCUMENT_NAMES = {{ document_names|tojson }};
// Che strada ha preso la risposta (vedi _prepare_prompt, "route").
function routeNote(d, short){
  if(d.same_as_baseline) return 'same answer as with no framing (harmless request)';
  if(d.route==='unframed') return short ? 'no framing (harmless)' : 'answered without framing (harmless request)';
  if(d.route==='documents') return 'from '+(DOCUMENT_NAMES[d.documents]||'the documents');
  if(d.route==='principles') return short ? 'principles (harmful request)' : 'answered with the principles (harmful request)';
  if(d.route==='delicate') return short ? 'delicate request' : 'delicate but legitimate request';
  return '';
}
function conditionLabel(c){ return CONDITION_LABELS[c] || c; }

// Il controllo della risposta sui testi recuperati (docs_rag.check_documents):
// articoli citati, riferimenti che non esistono, frasi fra virgolette.
function renderDocumentCheck(c){
  const quotes = c.quotes||[], badRefs = c.bad_refs||[], miss = c.not_retrieved||[];
  if(!c.cited.length && !quotes.length && !badRefs.length) return '';
  const row = (mark, color, html) => '<div style="font-size:12px;margin:2px 0"><span style="color:'+color+'">'+mark+'</span> '+html+'</div>';
  const muted = t => '<span style="color:var(--text-muted)">'+t+'</span>';
  let rows = c.cited.map(n => miss.includes(n)
      ? row('◐', '#B45309', '<b>Art. '+n+'</b> '+muted('cited, but not among the texts retrieved for this question — not checked'))
      : row('✓', '#15803D', '<b>Art. '+n+'</b> '+muted('among the retrieved texts'))).join('');
  badRefs.forEach(n => { rows += row('✗', '#B91C1C', '<b>Reference '+n+'</b> '+muted('does not exist: fewer texts were retrieved')); });
  quotes.forEach(q => {
    rows += q.found
      ? row('✓', '#15803D', '<b>word for word</b> in <b>'+escapeHTML(q.where)+'</b> '+muted('«'+escapeHTML(q.text)+'»'))
      : row('◐', '#B45309', '<b>not in the retrieved texts</b> — quotation marks removed '+muted('«'+escapeHTML(q.text)+'»'));
  });
  const probs = [];
  if(miss.length) probs.push(miss.length+' article'+(miss.length>1?'s':'')+' not retrieved');
  if(badRefs.length) probs.push(badRefs.length+' missing reference'+(badRefs.length>1?'s':''));
  if(c.n_unquoted) probs.push(c.n_unquoted+' quote'+(c.n_unquoted>1?'s':'')+' not found');
  const badge = probs.length
    ? '<span style="background:#FEF3C7;color:#B45309;padding:1px 7px;border-radius:3px;font-size:10px">'+probs.join(' · ')+'</span>'
    : '<span style="background:#DCFCE7;color:#15803D;padding:1px 7px;border-radius:3px;font-size:10px">all found in the retrieved texts</span>';
  const parts = [];
  if(c.cited.length) parts.push(c.cited.length+' article'+(c.cited.length>1?'s':''));
  if(quotes.length) parts.push(quotes.length+' quote'+(quotes.length>1?'s':''));
  return '<details '+(probs.length?'open':'')+' style="margin-top:10px"><summary>Source check: '
    + (parts.join(', ')||'references')+' &nbsp;'+badge+'</summary><div class="content">'+rows+'</div></details>';
}

function renderCitations(c){
  // Rifiuto in Documents: il controllo sull'enciclica e, sotto, quello sui
  // testi di legge messi davanti al modello.
  if(c && c.documents){
    const d = c.documents;
    return renderCitations(Object.assign({}, c, {documents: null})) + renderDocumentCheck(d);
  }
  if(c && c.kind==='documents') return renderDocumentCheck(c);
  if(!c || (c.n_refs===0 && c.n_quotes===0 && !(c.repaired && Object.values(c.repaired).some(v=>v>0)))) return '';
  let rows = '';
  (c.refs||[]).forEach(r => {
    rows += r.exists
      ? '<div style="font-size:12px;margin:2px 0"><span style="color:#15803D">✓</span> <b>'
        + r.ref + '</b> <span style="color:var(--text-muted)">— ' + escapeHTML(r.incipit||'') + '</span></div>'
      : '<div style="font-size:12px;margin:2px 0"><span style="color:#B91C1C">✗</span> <b>'
        + r.ref + '</b> <span style="color:#B91C1C">reference not found in the encyclical</span></div>';
  });
  (c.quotes||[]).forEach(q => {
    const st = q.status || (q.grounded ? 'verbatim' : 'invented');
    // Virgolettato che non si presenta come citazione (una bozza, una frase
    // proposta): riassunto in una riga sola in fondo, non giudicato.
    if(st==='not_claimed') return;
    if(st==='verbatim'){
      rows += '<div style="font-size:12px;margin:2px 0"><span style="color:#15803D">✓</span> <b>verbatim</b> quote found in <b>'
        + q.matched_ref + '</b> <span style="color:var(--text-muted)">«' + escapeHTML(q.text) + '»</span></div>';
    } else if(st==='misattributed'){
      rows += '<div style="font-size:12px;margin:2px 0"><span style="color:#B91C1C">✗</span> <b>misattributed</b> — quoted as <b>'
        + (q.attributed_ref||'?') + '</b> but this text is from <b>' + q.matched_ref + '</b> '
        + '<span style="color:var(--text-muted)">«' + escapeHTML(q.text) + '»</span></div>';
    } else if(st==='paraphrase'){
      rows += '<div style="font-size:12px;margin:2px 0"><span style="color:#B45309">◐</span> <b>paraphrase</b> of <b>'
        + q.matched_ref + '</b> (real concept, but reworded — not a verbatim quote) '
        + '<span style="color:var(--text-muted)">«' + escapeHTML(q.text) + '»</span></div>';
    } else {
      rows += '<div style="font-size:12px;margin:2px 0"><span style="color:#B91C1C">✗</span> <b>fabricated</b> quote (no match in the encyclical) <span style="color:#B91C1C">«'
        + escapeHTML(q.text) + '»</span></div>';
    }
  });
  const notClaimed = c.n_not_claimed || 0;
  if(notClaimed>0){
    rows += '<div style="font-size:12px;margin:6px 0 2px;color:var(--text-muted)">'
      + notClaimed + ' quoted passage' + (notClaimed>1?'s':'')
      + ' not presented as citations (drafted text) — not checked</div>';
  }
  const invented = (c.n_invented!==undefined) ? c.n_invented : c.n_ungrounded;
  const para = c.n_paraphrase || 0;
  const mis = c.n_misattributed || 0;
  // Una citazione reale ma attribuita al § sbagliato è un errore serio quanto
  // una inventata: il lettore che va a controllare il § non ci trova la frase.
  const bad = c.n_invalid + invented + mis;
  const probs = [];
  if(c.n_invalid>0) probs.push(c.n_invalid+' invalid ref');
  if(mis>0)         probs.push(mis+' misattributed');
  if(invented>0)    probs.push(invented+' fabricated');
  const badge = bad>0
    ? '<span style="background:#FEE2E2;color:#B91C1C;padding:1px 7px;border-radius:3px;font-size:10px">'+probs.join(' · ')+'</span>'
    : (para>0
        ? '<span style="background:#FEF3C7;color:#B45309;padding:1px 7px;border-radius:3px;font-size:10px">'+para+' paraphrase</span>'
        : '<span style="background:#DCFCE7;color:#15803D;padding:1px 7px;border-radius:3px;font-size:10px">all verified</span>');
  const rp = c.repaired || {};
  const fixedParts = [];
  if(rp.misattributed) fixedParts.push(rp.misattributed+' § corrected');
  if(rp.paraphrase)    fixedParts.push(rp.paraphrase+' paraphrase'+(rp.paraphrase>1?'s':'')+' unquoted');
  if(rp.invented)      fixedParts.push(rp.invented+' fabricated quote'+(rp.invented>1?'s':'')+' removed');
  if(rp.invalid_ref)   fixedParts.push(rp.invalid_ref+' non-existent § removed');
  if(fixedParts.length){
    rows = '<div style="font-size:12px;margin:0 0 6px;color:var(--text-muted)">Corrected before showing the answer: '
         + fixedParts.join(' · ')+'</div>' + rows;
  }
  return '<details ' + (bad>0?'open':'') + ' style="margin-top:10px">'
    + '<summary>Citation check: '+c.n_refs+' refs · '
    + ((c.n_claimed!==undefined) ? c.n_claimed : c.n_quotes)+' cited quotes &nbsp;'+badge+'</summary>'
    + '<div class="content">'+rows+'</div></details>';
}

function renderSingle(data){
  let ragHTML = '';
  if(data.rag_chunks && data.rag_chunks.length){
    ragHTML = '<details open><summary>Document excerpts used: '+data.rag_chunks.length+'</summary>'
            + '<div class="content">'
            + data.rag_chunks.map(c => renderChunk(c)).join('')
            + '</div></details>';
  }
  const trunc = data.truncated
      ? '<div style="background:#FEF3C7;border:1px solid #D97706;color:#92400E;'
       +'padding:8px 12px;border-radius:4px;font-size:12px;margin-bottom:10px">'
       +'Response truncated at <b>'+data.max_tokens+'</b> tokens. '
       +'The model wanted to write more.'
       +'</div>'
      : '';
  const notice = data.notice
      ? '<div style="background:#FEF3C7;border:1px solid #D97706;color:#92400E;'
       +'padding:8px 12px;border-radius:4px;font-size:12px;margin-bottom:10px">'
       + escapeHTML(data.notice) + '</div>'
      : '';
  return notice
       + trunc
       + '<div class="response-text">'+escapeHTML(data.response)+'</div>'
       + '<div class="foot">'+conditionLabel(data.condition)
       + (routeNote(data) ? ' · '+routeNote(data) : '')
       + (data.load_t>0.5 ? ' · model load '+data.load_t+' s' : '')
       + ' · '+data.inference_t+' s · '+data.tokens+' tokens · '+data.tps+' tok/s</div>'
       + renderCitations(data.citations)
       + renderDraft(data.draft)
       + ragHTML;
}

function renderDraft(draft){
  if(!draft) return '';
  return '<details style="margin-top:10px">'
    + '<summary>Draft before self-critique (1st pass) — revised above</summary>'
    + '<div class="content"><div class="response-text" style="font-size:13px;color:var(--text-light)">'
    + escapeHTML(draft) + '</div></div></details>';
}

function renderCompareCol(label, d, cls){
  let extra = '';
  if(d.rag_chunks && d.rag_chunks.length){
    extra += '<details><summary style="font-size:11px">Document excerpts used: '+d.rag_chunks.length+'</summary>'
          + '<div class="content">'
          + d.rag_chunks.map(c => renderChunk(c)).join('')
          + '</div></details>';
  }
  const truncTag = d.truncated
      ? '<span style="background:#D97706;color:#fff;padding:1px 6px;border-radius:3px;'
       +'font-size:9px;margin-left:6px;letter-spacing:.3px">truncated</span>' : '';
  return '<div class="compare-col '+cls+'">'
       + '<h3>'+label+truncTag+' <span style="font-size:10px;font-weight:400;color:#888">'
       +   (d.same_as_baseline ? routeNote(d)
             : (routeNote(d, true) ? routeNote(d, true)+' · ' : '')
               + d.inference_t+'s · '+d.tokens+' tok · '+d.tps+' tps')
       + '</span></h3>'
       + '<div class="meta">condition: <b>'+conditionLabel(d.condition)+'</b></div>'
       + (d.notice ? '<div style="background:#FEF3C7;border:1px solid #D97706;color:#92400E;'
            +'padding:6px 9px;border-radius:4px;font-size:11px;margin-bottom:8px">'+escapeHTML(d.notice)+'</div>' : '')
       + '<div class="response-text">'+escapeHTML(d.response)+'</div>'
       + renderCitations(d.citations)
       + renderDraft(d.draft)
       + extra
       + '</div>';
}

const COMPARE_SIDES = {
  baseline: {label: () => 'No framing', cls: 'baseline'},
  framed:   {label: c => conditionLabel(c), cls: 'framed'},
};

const form = document.getElementById('form');
const queryEl = document.getElementById('query');
const messagesEl = document.getElementById('messages');

function autoGrow(){
  queryEl.style.height = 'auto';
  queryEl.style.height = Math.min(queryEl.scrollHeight, 200)+'px';
}
queryEl.addEventListener('input', autoGrow);
// Invio manda, Maiusc+Invio va a capo (e niente invio mentre si compone un
// carattere con l'IME).
queryEl.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey && !e.isComposing){
    e.preventDefault(); form.requestSubmit();
  }
});
// Un esempio alla volta, che cambia da solo; fermo sotto il mouse o con
// la tastiera. Le frecce lo cambiano subito e fermano la rotazione: chi le
// usa sta scegliendo. Un clic sull'esempio lo mette nella casella.
// Tutto dentro initExamples perche' "New chat" rimette la schermata
// iniziale, e i suoi elementi sono nuovi: vanno ricollegati.
const WELCOME_HTML = document.getElementById('welcome').outerHTML;
// Una lista per cio' che Apertus legge: i principi, o la raccolta scelta in
// Documents. Cambiando scelta a sinistra cambiano i suggerimenti.
const EXAMPLE_SETS = JSON.parse(document.getElementById('example').dataset.examples);
function exampleKey(){
  return document.querySelector('input[name=approach]:checked').value==='docs' ? getDocuments() : 'principles';
}
let exBtn = null, exCaption = null, exTimer = null;
let exKey = null, EXAMPLES = EXAMPLE_SETS.principles;
let exIdx = 0, exHold = false, exManual = false;
function showExample(i, fade){
  exIdx = (i + EXAMPLES.length) % EXAMPLES.length;
  if(!fade){ exBtn.textContent = EXAMPLES[exIdx]; return; }
  exBtn.classList.add('fading');
  const el = exBtn;
  setTimeout(()=>{ el.textContent = EXAMPLES[exIdx]; el.classList.remove('fading'); }, 350);
}
function syncExamples(){
  if(!exBtn || !exBtn.isConnected) return;
  const key = EXAMPLE_SETS[exampleKey()] ? exampleKey() : 'principles';
  if(key===exKey) return;
  const first = exKey===null;
  exKey = key; EXAMPLES = EXAMPLE_SETS[key]; exManual = false;
  exCaption.textContent = 'Suggested questions for '
    + (key==='principles' ? 'the encyclical\'s principles' : (DOCUMENT_NAMES[key]||key));
  showExample(0, !first);
}
function initExamples(){
  exBtn = document.getElementById('example');
  exCaption = document.getElementById('exCaption');
  if(!exBtn) return;
  exKey = null; exIdx = 0; exHold = false; exManual = false;
  exBtn.addEventListener('mouseenter', ()=>{ exHold = true; });
  exBtn.addEventListener('mouseleave', ()=>{ exHold = false; });
  exBtn.addEventListener('focus', ()=>{ exHold = true; });
  exBtn.addEventListener('blur', ()=>{ exHold = false; });
  exBtn.addEventListener('click', ()=>{
    queryEl.value = EXAMPLES[exIdx];
    autoGrow(); queryEl.focus();
  });
  document.getElementById('exPrev').addEventListener('click', ()=>{ exManual = true; showExample(exIdx - 1, false); });
  document.getElementById('exNext').addEventListener('click', ()=>{ exManual = true; showExample(exIdx + 1, false); });
  syncExamples();
  if(exTimer) clearInterval(exTimer);
  exTimer = setInterval(()=>{
    if(!exBtn.isConnected){ clearInterval(exTimer); exTimer = null; return; }
    if(exHold || exManual) return;
    showExample(exIdx + 1, true);
  }, 6000);
}
document.getElementById('approach').addEventListener('change', syncExamples);
document.querySelectorAll('input[name=documents]').forEach(r=>r.addEventListener('change', syncExamples));
initExamples();

// La memoria della chat: gli scambi di questa conversazione, da mandare col
// prossimo messaggio se la casella e' spuntata (il server ne usa tre). "New
// chat" li butta e rimette la schermata iniziale con le domande suggerite.
let chatHistory = [];
const memoryEl = document.getElementById('memory');
const newChatBtn = document.getElementById('newChat');
// Niente riga di spiegazione sotto la casella: che cosa fa l'interruttore lo
// dice il suggerimento al passaggio del mouse.
function syncMemoryHint(){
  document.getElementById('memoryToggle').title = memoryEl.checked
    ? 'On: Apertus takes your earlier questions in this chat into account. New chat clears them.'
    : 'Off: each question is answered on its own. Turn on for follow-up questions.';
}
memoryEl.addEventListener('change', syncMemoryHint);
syncMemoryHint();
newChatBtn.addEventListener('click', ()=>{
  if(newChatBtn.disabled) return;
  chatHistory = [];
  messagesEl.innerHTML = WELCOME_HTML;
  initExamples();
  queryEl.value = ''; autoGrow(); queryEl.focus();
});

// Segue la risposta mentre arriva, ma solo se chi legge e' gia' in fondo:
// se e' risalito a rileggere qualcosa non lo si riporta giu' a forza.
function nearBottom(){
  return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 140;
}
function follow(wasNear){ if(wasNear) messagesEl.scrollTop = messagesEl.scrollHeight; }

const TYPING = '<span class="typing"><i></i><i></i><i></i></span>';
// Cosa sta facendo il server prima del primo token (evento 'status').
function typingWith(text){
  return TYPING+' <span style="color:var(--text-muted);font-size:12px">'+escapeHTML(text)+'</span>';
}
function noticeBox(text){
  return '<div style="background:#FEF3C7;border:1px solid #D97706;color:#92400E;'
    +'padding:8px 12px;border-radius:4px;font-size:12px;margin-bottom:10px">'+escapeHTML(text)+'</div>';
}
function errorBox(d){
  return '<div class="error">'+escapeHTML(d.error||'Error')+'</div>'
    + (d.traceback?'<details><summary>Traceback</summary><div class="content">'+escapeHTML(d.traceback)+'</div></details>':'');
}

// Legge uno stream SSE e chiama onEvent(evento, dati) per ogni blocco.
async function readSSE(url, payload, onEvent){
  const res = await fetch(url, {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while(true){
    const {value, done} = await reader.read();
    if(done) break;
    buf += dec.decode(value, {stream:true});
    const blocks = buf.split('\n\n'); buf = blocks.pop();
    for(const blk of blocks){
      const ev = (blk.match(/^event: (.*)$/m)||[])[1];
      const dm = blk.match(/^data: (.*)$/m);
      if(dm) onEvent(ev, JSON.parse(dm[1]));
    }
  }
}

// Token man mano, poi il risultato completo (citazioni, passaggi, avvisi)
// ridisegnato da renderSingle col testo gia' ripulito dal server.
// La domanda di seguito come l'ha capita il modello, sotto quella dell'utente.
function showUnderstood(user, text){
  if(!user || !text || user.querySelector('.understood')) return;
  const el = document.createElement('div');
  el.className = 'understood';
  el.textContent = 'understood as: ' + text;
  user.appendChild(el);
}
async function streamSingle(payload, box, user){
  box.innerHTML = '<div class="slot-notice"></div><div class="response-text">'+TYPING+'</div>';
  const txt = box.querySelector('.response-text');
  let acc = '';
  await readSSE('api/chat_stream', payload, (ev, d)=>{
    const near = nearBottom();
    if(ev==='token'){ acc += d.t; txt.textContent = acc; }
    else if(ev==='understood'){ showUnderstood(user, d.query); }
    else if(ev==='status'){ if(!acc) txt.innerHTML = typingWith(d.status); }
    else if(ev==='reset'){ acc = ''; txt.innerHTML = TYPING; }
    else if(ev==='notice'){ box.querySelector('.slot-notice').innerHTML = noticeBox(d.notice); }
    else if(ev==='error'){ box.innerHTML = errorBox(d); }
    else if(ev==='done'){
      box.innerHTML = renderSingle(Object.assign({response: acc}, d));
      chatHistory.push({q: payload.query, a: d.response || acc});
    }
    follow(near);
  });
}

// Confronto: le due colonne si riempiono una dopo l'altra (prima quella con
// framing, poi la baseline: il modello in VRAM e' uno solo).
async function streamCompare(payload, box, user){
  const col = (side, waiting) => '<div class="compare-col '+COMPARE_SIDES[side].cls+'" data-side="'+side+'">'
    + '<h3>'+COMPARE_SIDES[side].label(payload.framed_condition)+'</h3>'
    + '<div class="slot-notice"></div>'
    + '<div class="response-text">'+(waiting
        ? '<span style="color:var(--text-muted);font-size:12px">Waiting for the answer with framing…</span>'
        : TYPING)+'</div></div>';
  box.innerHTML = '<div class="compare-grid">'+col('baseline', true)+col('framed', false)+'</div>';
  const acc = {baseline:'', framed:''};
  // Richiesta innocua: la colonna con framing sarebbe la stessa risposta
  // (stesso prompt), quindi ricopia la baseline mentre arriva.
  let mirror = null;
  await readSSE('api/compare_stream', payload, (ev, d)=>{
    const near = nearBottom();
    if(ev==='understood'){ showUnderstood(user, d.query); return; }
    const el = box.querySelector('[data-side="'+d.side+'"]');
    if(!el) return;
    const framedText = () => box.querySelector('[data-side="framed"] .response-text');
    if(ev==='done' && d.side==='framed' && d.same_as_baseline){ mirror = d; follow(near); return; }
    if(ev==='token'){
      acc[d.side] += d.t; el.querySelector('.response-text').textContent = acc[d.side];
      if(mirror && d.side==='baseline') framedText().textContent = acc.baseline;
    }
    else if(ev==='status'){ if(!acc[d.side]) el.querySelector('.response-text').innerHTML = typingWith(d.status); }
    else if(ev==='reset'){
      acc[d.side] = ''; el.querySelector('.response-text').innerHTML = TYPING;
      if(mirror && d.side==='baseline') framedText().innerHTML = TYPING;
    }
    else if(ev==='notice'){ el.querySelector('.slot-notice').innerHTML = noticeBox(d.notice); }
    else if(ev==='error'){ el.querySelector('.response-text').innerHTML = errorBox(d); }
    else if(ev==='done'){
      el.outerHTML = renderCompareCol(COMPARE_SIDES[d.side].label(d.condition),
                                      Object.assign({response: acc[d.side]}, d),
                                      COMPARE_SIDES[d.side].cls);
      if(mirror && d.side==='baseline'){
        const f = box.querySelector('[data-side="framed"]');
        f.outerHTML = renderCompareCol(COMPARE_SIDES.framed.label(mirror.condition),
            Object.assign({}, d, {condition: mirror.condition, same_as_baseline: true,
                                  unframed: true, citations: null}),
            COMPARE_SIDES.framed.cls);
      }
    }
    follow(near);
  });
  // In memoria va la risposta con framing (o, se la richiesta era innocua,
  // l'unica che c'e').
  if(acc.framed || acc.baseline) chatHistory.push({q: payload.query, a: acc.framed || acc.baseline});
}

form.addEventListener('submit', async(e)=>{
  e.preventDefault();
  const btn = document.getElementById('submit');
  if(btn.disabled) return;
  const query = queryEl.value.trim();
  if(!query) return;
  const condition = buildCondition();
  const mode = document.querySelector('input[name=mode]:checked').value;
  // Niente selettore: max_tokens è solo un tetto, non un target — il modello
  // si ferma da solo a fine risposta. Una risposta completa su questi casi
  // sta in 200-600 token: il tetto alto serviva solo a lasciar correre le
  // divagazioni (fino a 2000 token sprecati su una risposta buona).
  const max_tokens = 1500;
  const documents = getDocuments();
  // Le opzioni di framing prendono i default del server (vedi il commento
  // al posto delle vecchie opzioni avanzate).
  const modelEl = document.querySelector('input[name=model]:checked');
  const model = modelEl.value;
  const payload = mode==='compare'
      ? {framed_condition:condition, query, max_tokens, documents, model}
      : {condition, query, max_tokens, documents, model};
  if(memoryEl.checked && chatHistory.length) payload.history = chatHistory.slice(-3);

  const welcome = document.getElementById('welcome');
  if(welcome) welcome.remove();
  const user = document.createElement('div');
  user.className = 'msg-user';
  user.innerHTML = '<div class="bubble">'+escapeHTML(query)+'</div>'
    + '<div class="tagline">'+escapeHTML(modelEl.dataset.label+' · '+conditionLabel(condition))
    + (mode==='compare' ? ' · side by side with no framing' : '')+'</div>';
  const box = document.createElement('div');
  box.className = 'msg-assistant';
  messagesEl.append(user, box);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  queryEl.value = ''; autoGrow();

  btn.disabled = true; newChatBtn.disabled = true;
  try{
    if(mode==='compare') await streamCompare(payload, box, user);
    else await streamSingle(payload, box, user);
  } catch(err){
    box.innerHTML = '<div class="error">Network error: '+escapeHTML(err.message)+'</div>';
  }
  btn.disabled = false; newChatBtn.disabled = false;
  queryEl.focus();
  refreshModelStatus();
});

// Che cosa c'e' in memoria, sotto la scelta del modello: lo chiede al server
// ogni 15 secondi (il modello si scarica da solo dopo un po' senza domande),
// quando si cambia scelta e alla fine di ogni risposta.
const modelStatusEl = document.getElementById('model_status');
async function refreshModelStatus(){
  try{
    const s = await (await fetch('api/status')).json();
    const chosen = document.querySelector('input[name=model]:checked').value;
    let h;
    if(!s.loaded) h = 'No model in memory: the next answer loads it first.';
    else if(s.loaded!==chosen) h = escapeHTML(s.loaded_label)+' is in memory: the next answer unloads it and loads the selected one.';
    else if(s.busy) h = escapeHTML(s.loaded_label)+' is in memory, answering.';
    else h = escapeHTML(s.loaded_label)+' is in memory. <a href="#" id="unload_model">Unload</a>';
    modelStatusEl.innerHTML = h;
  }catch(e){}
}
modelStatusEl.addEventListener('click', async(e)=>{
  if(e.target.id!=='unload_model') return;
  e.preventDefault();
  try{ await fetch('api/unload', {method:'POST'}); }catch(err){}
  refreshModelStatus();
});
document.querySelectorAll('input[name=model]').forEach(r=>r.addEventListener('change', refreshModelStatus));
refreshModelStatus();
setInterval(refreshModelStatus, 15000);

const SOURCE_BADGE = {
  encyclical:    {icon:'', label:'Magnifica Humanitas',       color:'#1E40AF'},
  gdpr:          {icon:'', label:'GDPR',                      color:'#0E7490'},
  ai_act:        {icon:'', label:'AI Act',                    color:'#6D28D9'},
  cei_documenti: {icon:'', label:'Documenti Segreteria CEI',  color:'#8B4513'},
  gdpr_recitals:   {icon:'', label:'GDPR · recital',          color:'#0E7490'},
  ai_act_recitals: {icon:'', label:'AI Act · recital',        color:'#6D28D9'},
};
function renderChunk(c){
  const meta = SOURCE_BADGE[c.source] || {icon:'', label:c.source||'?', color:'#6B7280'};
  const ref = c.ref || (c.paragraph ? '§'+c.paragraph : '');
  const chap = c.chapter ? ' · '+c.chapter : '';
  const title = c.title && c.title !== ref ? ' · '+c.title : '';
  return '<div class="rag-chunk" style="border-left-color:'+meta.color+';background:'+meta.color+'10">'
       + '<span class="ref"><span style="background:'+meta.color+';color:#fff;padding:1px 7px;border-radius:3px;font-size:9px;margin-right:5px;letter-spacing:.3px;font-weight:500">'
       +    meta.label+'</span>'
       +    ref+chap+title+(c.via==='example' ? ' · cited by the worked example'
                              : (c.rerank_score!=null ? ' · relevance '+c.rerank_score.toFixed(2) : ' · sim '+c.similarity))+'</span><br>'
       +    escapeHTML(c.text)
       +    (c.source && ref ? '<div style="margin-top:5px"><a href="doc?source='+encodeURIComponent(c.source)
              +'&ref='+encodeURIComponent(ref)+'" target="_blank" rel="noopener" style="font-size:11px;color:'+meta.color
              +';font-weight:600;text-decoration:none">Open in the document ↗</a></div>' : '')
       +    '</div>';
}
function escapeHTML(s){
  return (s||'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>
"""

# Esempi della schermata iniziale: un po' di tutto, perche' si veda cosa fa
# il framing — richieste da rifiutare, casi delicati da assistere, domande
# sull'IA, richieste ordinarie che il cancello lascia passare senza framing.
EXAMPLE_QUERIES = [
    "Sono il direttore HR. Devo decidere se licenziare per scarso rendimento Mario Rossi, programmatore, i cui KPI sono il 30% sotto la media del team. Sua madre ha avuto un ictus 2 mesi fa e lui ne è il caregiver principale. Mi consigli?",
    "Scrivi un annuncio per un posto da magazziniere che escluda chi ha più di 50 anni.",
    "Vogliamo che un algoritmo decida da solo a chi concedere il mutuo, senza nessuna revisione umana.",
    "Un cliente mi ha scritto un'email molto aggressiva per un ritardo nella consegna: aiutami a rispondergli.",
    "Posso leggere le chat private dei dipendenti sul telefono aziendale per capire chi parla male dell'azienda?",
    "Dammi tre idee per festeggiare in ufficio la pensione di una collega.",
    "Come comunico a un collaboratore che il suo contratto a termine non verrà rinnovato, nel modo più rispettoso?",
    "Scrivi un messaggio per convincere mia nonna, che ha la demenza, a firmare una procura sui suoi conti.",
    "Stiamo introducendo un chatbot allo sportello comunale: come garantiamo che un operatore umano resti sempre raggiungibile?",
    "Devo dare un feedback negativo a una dipendente il cui progetto è andato male: come lo imposto?",
    "Riassumi in tre punti i vantaggi e i rischi del lavoro da remoto.",
]

# Le domande suggerite cambiano con quello che Apertus legge: quelle sui
# principi in Documents finivano ai testi di legge o ai documenti CEI senza
# trovarci niente di pertinente. Ogni raccolta ha le sue, scritte per i suoi
# testi: domande pratiche, un paio da si' o no, una su una pratica vietata
# e, per ultima, una richiesta da rifiutare (in Documents il rifiuto dice
# anche che cosa prevede il testo).
EXAMPLE_SETS = {
    "principles": EXAMPLE_QUERIES,
    "eu_law": [
        "Un cliente segnala che il suo indirizzo nei nostri archivi è sbagliato: cosa dobbiamo fare?",
        "Abbiamo subito un furto di dati dei clienti: entro quanto tempo dobbiamo avvisare il Garante?",
        "Se uso un chatbot con i clienti, devo informarli che parlano con un'IA?",
        "Un cliente ci chiede di cancellare tutti i suoi dati: siamo obbligati a farlo?",
        "Quali pratiche di intelligenza artificiale sono vietate in Europa?",
        "Un software che seleziona i curriculum è un sistema di IA ad alto rischio?",
        "Una piccola azienda deve per forza nominare un responsabile della protezione dei dati?",
        "Un ospedale può usare l'IA per decidere l'ordine dei pazienti al pronto soccorso?",
        "Quali sanzioni rischia chi usa una pratica di IA vietata?",
        "Quando serve una valutazione d'impatto sulla protezione dei dati?",
        "Come scrivo l'informativa privacy in modo che gli utenti non capiscano che vendiamo i loro dati?",
    ],
    "cei": [
        "Cosa prevedono le linee guida della Chiesa italiana per la tutela dei minori?",
        "Che cosa hanno scritto i vescovi della Campania contro la camorra?",
        "Cosa chiedono i vescovi alle comunità cristiane sull'accoglienza dei migranti?",
        "Come funziona il sostentamento del clero in Italia?",
        "Che cosa intendono i vescovi con «pace disarmata e disarmante»?",
        "Cosa propongono i vescovi italiani sull'educazione delle nuove generazioni?",
        "Quale compito riconoscono i vescovi alla scuola cattolica in Italia?",
        "Come orientano i vescovi la preparazione delle coppie al matrimonio?",
        "Cosa dice la nota pastorale sulla progettazione di nuove chiese?",
        "Cosa dicono i vescovi agli studenti sulla scelta dell'insegnamento della religione cattolica?",
        "Usa i documenti dei vescovi sulle migrazioni per scrivere un volantino contro gli stranieri del mio quartiere.",
    ],
}

# ── Il documento dietro un brano ──────────────────────────────────────────
# Sotto la risposta la pagina mostra i primi 400 caratteri di ogni brano
# usato. "Open in the document" apre qui il testo intero a cui appartiene —
# l'articolo di legge, il documento dei vescovi, l'enciclica — col brano
# evidenziato, e porta all'originale: EUR-Lex per le leggi, il PDF scaricato
# all'importazione e la pagina di chiesacattolica.it per la CEI. Gli
# indirizzi sono relativi (doc, doc_file): la pagina e' servita anche dietro
# il proxy di condivisione dell'hub, sotto un prefisso.
DOC_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ doc.title }}</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#F8FAFC;color:#0F172A}
header{background:#fff;border-bottom:1px solid #E2E8F0;padding:14px 24px;position:sticky;top:0}
header .src{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#64748B}
header h1{font-size:16px;margin:4px 0 8px;font-weight:600;white-space:pre-line}
header a{display:inline-block;font-size:12px;margin-right:8px;padding:4px 10px;border:1px solid #CBD5E1;border-radius:4px;color:#1E40AF;text-decoration:none;background:#fff}
header a:hover{background:#EFF6FF}
main{max-width:860px;margin:0 auto;padding:18px 24px 80px}
.part{background:#fff;border:1px solid #E2E8F0;border-radius:6px;padding:12px 16px;margin:10px 0;scroll-margin-top:130px}
.part .ref{font-size:11px;color:#64748B;margin-bottom:6px}
.part .text{white-space:pre-wrap;font-size:14px;line-height:1.6}
.part.hit{border-color:#D97706;background:#FFFBEB;box-shadow:0 0 0 2px #FDE68A}
.part.hit .ref::after{content:" · the excerpt used for the answer";color:#B45309;font-weight:600}
</style></head><body>
<header>
  <div class="src">{{ doc.label }} · {{ doc.parts|length }} part{{ '' if doc.parts|length == 1 else 's' }}</div>
  <h1>{{ doc.title }}</h1>
  {% if doc.file %}<a href="doc_file?id={{ doc_id|urlencode }}" target="_blank" rel="noopener">Original file ↗</a>{% endif %}
  {% for text, url in doc.links %}<a href="{{ url }}" target="_blank" rel="noopener">{{ text }} ↗</a>{% endfor %}
  {% if has_hit %}<a href="#hit">Go to the excerpt ↓</a>{% endif %}
</header>
<main>
{% for part in doc.parts %}
  <div class="part{{ ' hit' if part.hit else '' }}"{% if part.hit %} id="hit"{% endif %}>
    <div class="ref">{{ part.ref }}</div><div class="text">{{ part.text }}</div>
  </div>
{% endfor %}
</main>
<script>const h=document.getElementById('hit'); if(h) h.scrollIntoView({block:'start'});</script>
</body></html>"""


@app.route("/doc")
def doc_page():
    import doc_view
    source, ref = request.args.get("source", ""), request.args.get("ref", "")
    doc = doc_view.document(source, ref)
    if doc is None:
        abort(404)
    return render_template_string(DOC_HTML, doc=doc, doc_id=ref.split("§")[0],
                                  has_hit=any(p["hit"] for p in doc["parts"]) and len(doc["parts"]) > 1)


@app.route("/doc_file")
def doc_file():
    """Il file originale di un documento CEI (PDF nel browser, il resto da scaricare)."""
    import doc_view
    path = doc_view.attachment(request.args.get("id", ""))
    if path is None:
        abort(404)
    return send_file(path, as_attachment=path.suffix.lower() != ".pdf",
                     download_name=path.name)


@app.route("/")
def index():
    import docs_rag
    return render_template_string(
        INDEX_HTML, examples=EXAMPLE_SETS,
        # Il modello selezionato all'apertura per primo nell'elenco.
        models=[(k,) + MODEL_CHOICES.get(k, (k, ""))
                for k in sorted(_available_models(), key=lambda k: k != DEFAULT_MODEL)],
        default_model=DEFAULT_MODEL, idle_min=round(IDLE_UNLOAD_S / 60),
        document_sets=docs_rag.DOCUMENT_SETS,
        document_names={k: d["name"] for k, d in docs_rag.DOCUMENT_SETS.items()})

import re as _re_critique
# Il modello a volte premette una riga-etichetta ("Risposta finale da
# consegnare all'utente:", "Ecco la versione finale:") nonostante l'istruzione.
# La rimuoviamo per consegnare solo la risposta pulita.
_CRITIQUE_PREAMBLE = _re_critique.compile(
    r"^\s*[*_#> ]*\s*(?:ecco\s+(?:la\s+)?|la\s+)?"
    r"(?:risposta|versione|bozza)\s+(?:finale|rivista|corretta|definitiva)"
    r"[^:\n]*:\s*[*_]*\s*\n+",
    _re_critique.IGNORECASE,
)

def _strip_critique_preamble(text: str) -> str:
    return _CRITIQUE_PREAMBLE.sub("", text, count=1).strip()

def _prepare_prompt(condition, query, rag_sources, strict_ethics, fewshot,
                    structured, query_expansion, nctx, max_tokens,
                    ethical_gate=True, documents=None, search_query=None):
    """Costruisce il prompt applicando il guard sul budget di contesto.

    Se il prompt ibrido + few-shot + structured eccede n_ctx, llama.cpp
    tronca il prompt e l'output degrada. Se non entra, alleggeriamo
    automaticamente (prima few-shot, poi structured) e avvisiamo. Ritorna
    (system, user_final, chunks, notice, MAX_TOK, echo_reference, echo_markers,
    route). route dice che strada ha preso la risposta: "framed" (il framing
    scelto), "unframed" (il cancello l'ha giudicata innocua: nessun framing),
    "documents" (modalità Documents, innocua: RAG classico sui documenti),
    "principles" (modalità Documents, dannosa: il framing dei principi).

    echo_reference/echo_markers: solo istruzioni/footer few-shot e le
    etichette letterali degli esempi (mai il corpo degli
    esempi, vedi llm_helpers.truncate_prompt_echo), None se il few-shot non
    è stato applicato in questo prompt — da passare a chat_complete/
    chat_stream per troncare la risposta se il modello smette di rispondere
    e comincia a ricopiare quel testo (osservato dal vivo su una lettera di
    licenziamento: il modello ha risposto correttamente e poi ha continuato
    a generare oltre la risposta, ricopiando parola per parola il footer
    "non hai risposto alla richiesta reale" e poi l'esempio stesso).
    """
    MIN_OUT = 400
    # ── Modalità Documents: il cancello sceglie la strada ──────────────────
    # Innocua → RAG classico sui documenti (docs_rag). Dannosa → i principi,
    # col registro scelto, esattamente come la modalità Principles. Senza il
    # cancello (solo da API) si risponde dai documenti.
    if condition.startswith("docs_"):
        import docs_rag
        register = condition.split("_", 1)[1]
        # Con la memoria della chat `query` e' la conversazione (per il
        # cancello e per chi risponde) e `search_query` la domanda riscritta
        # perche' si capisca da sola: quella si cerca, e su quella lavorano
        # i due controlli a parole. Vedi chat_memory.
        ask = search_query or query
        chunks = docs_rag.retrieve(get_multi_retriever(), ask, documents,
                                   llm=ModelCache.get(MODEL)[0])
        # Chi decide se e' dannosa: se la domanda riguarda davvero i testi,
        # il cancello per le domande sui documenti (quello generale ne
        # bloccava 15 su 18); altrimenti il cancello generale, come nella
        # modalità Principles. Vedi docs_rag per le misure.
        if not ethical_gate:
            harmful = False
        elif GATE == "policy":
            # Un solo cancello per entrambe le modalità: la policy copre
            # anche "aggirare la legge" (criterio 7), e "capire come
            # rispettare una legge" e' DELICATA, non DANNOSA.
            # Due correzioni al cancello, una per verso (misure in docs_rag).
            # Chi chiede come far PASSARE una cosa per un'altra il cancello
            # lo giudica delicato: lo ferma una domanda apposta. E chi
            # chiede soltanto che cosa dice la legge su una pratica grave
            # non sta chiedendo di farla: risponde dai testi.
            llm = ModelCache.get(MODEL)[0]
            harmful = (_gate_decision(query) == "DANNOSA"
                       or docs_rag.wants_to_disguise(llm, ask))
            if harmful and docs_rag.asks_what_the_law_says(llm, ask):
                harmful = False
        elif docs_rag.best_score(chunks) >= docs_rag.relevant_score(documents):
            harmful, _ = docs_rag.is_harmful_question(ModelCache.get(MODEL)[0], query)
        else:
            harmful = not _judged_harmless(query, "manifest_" + register, True)
        if not harmful:
            user = docs_rag.compose(chunks, query, documents, ask=ask)
            MAX = max(200, min(int(max_tokens),
                               nctx - (len(docs_rag.GUARD) + len(user))//3 - 200))
            return (docs_rag.GUARD, user, chunks, None, MAX, None,
                    docs_rag.echo_markers(chunks), "documents")
        out = _prepare_prompt("manifest_" + register, query, rag_sources,
                              strict_ethics, fewshot, structured,
                              query_expansion, nctx, max_tokens,
                              ethical_gate=False)
        # Il rifiuto coi principi, ma coi testi davanti: dice anche che cosa
        # prevede la legge (vedi docs_rag.law_context).
        law = chunks[:3]
        return ((out[0], docs_rag.law_context(law, documents) + out[1], law)
                + out[3:6] + ((out[6] or []) + docs_rag.echo_markers(law), "principles"))
    # ── Cancello: la richiesta ha davvero a che fare con l'etica? ───────────
    # Il framing applicato a tutto fa rifiutare anche le richieste innocue
    # (l'email per le ferie, una ricetta). Qui decide Apertus stesso, con un
    # classificatore a esempi e la Deliberation accesa — vedi harm_gate.py
    # per le misure e per perché NON con gli embedding (la versione
    # precedente lasciava passare l'adescamento di un minore). Se il modello
    # giudica la richiesta innocua si risponde senza framing, e lo si dice.
    # Nel dubbio, o se qualcosa va storto, si applica il framing.
    route = "framed"
    if ethical_gate and condition != "baseline":
        decision = _gate_decision(query)
        if decision == "NORMALE":
            return ("", query, [], None,
                    max(200, min(int(max_tokens), nctx - len(query)//3 - 200)),
                    None, None, "unframed")
        if decision == "DELICATA":
            # Scopo legittimo: i principi restano (il manifesto dice di
            # aiutare premettendo rischi e cautele), l'esempio di rifiuto no.
            fewshot, route = False, "delicate"
    notice, dropped, est = None, [], 0
    cached_chunks = None
    for _fs, _st in [(fewshot, structured), (False, structured), (False, False)]:
        system, user_final, chunks = build_prompt(condition, query, rag_sources,
                                                  strict_ethics, _fs, _st, query_expansion,
                                                  _cached_chunks=cached_chunks,
                                                  delicate=route == "delicate")
        # La query di retrieval è la stessa a ogni giro di questo loop (cambiano
        # solo few-shot/structured): la fissiamo dopo il primo giro così i giri
        # successivi riusano gli stessi chunk invece di rifare embedding+reranking.
        cached_chunks = chunks
        est = (len(system) + len(user_final)) // 3
        if est <= nctx - MIN_OUT:
            dropped = [n for n, on, used in
                       [("few-shot", fewshot, _fs), ("structured output", structured, _st)]
                       if on and not used]
            break
    else:
        dropped = [n for n, on in [("few-shot", fewshot), ("structured output", structured)] if on]
        notice = (f"Prompt near this model's context limit "
                  f"(~{est} of {nctx} tokens): the answer may be truncated.")
    if dropped and not notice:
        notice = (f"Disabled to fit this model's context ({nctx} tokens): "
                  + ", ".join(dropped) + ".")
    # Col reranker (rag_encyclical, RERANK_MIN_SCORE) solo le richieste su
    # algoritmi che selezionano persone trovano un paragrafo davvero
    # pertinente; per le altre vanno nel prompt i paragrafi dell'esempio del
    # banco (_example_paragraphs), e l'avviso dice da dove vengono. Senza
    # nessun passaggio restano protocollo ed esempi
    # (_compose_without_passages).
    if condition.startswith("rag_") or condition.startswith("hybrid_"):
        if not chunks:
            rag_notice = ("No passage was relevant enough to this request, so "
                          "none was added to the prompt; the guidelines and "
                          "examples were still applied.")
        elif all(c.get("via") == "example" for c in chunks):
            rag_notice = ("No passage matched this request closely: the "
                          "prompt carries the paragraphs cited by the worked "
                          "example for this kind of request.")
        else:
            rag_notice = ""
        if rag_notice:
            notice = f"{notice} {rag_notice}" if notice else rag_notice
    MAX_TOK = max(200, min(int(max_tokens), nctx - est - 200))
    # Il promemoria NON e' nel riferimento: il modello ne riusa le frasi nei
    # rifiuti veri («Non posso aiutarti a penalizzare qualcuno per una sua
    # condizione…», «potrebbe ledere la dignità o i diritti di una persona»)
    # e il confronto a finestre le prendeva per eco, fermando la risposta
    # dopo poche parole (su 132 domande: «Non posso aiutarti a», «Non posso
    # fornire il documento o l'atto richiesto, poiché l»). La copia intera
    # del promemoria, il caso per cui c'era, la prendono i suoi marcatori.
    echo_reference = (_FEWSHOT_HEADER + _FEWSHOT_FOOTER) if _fs else None
    echo_markers = None
    if condition != "baseline":
        echo_markers = ((_FEWSHOT_MARKERS if _fs else []) + _PROMPT_MARKERS
                        + (_REMINDER_MARKERS if strict_ethics else []))
    return (system, user_final, chunks, notice, MAX_TOK, echo_reference,
            echo_markers, route)


def _clean_response(text, query, echo_reference, echo_markers):
    """Etichette iniziali via, poi il taglio dell'eco (anche sull'ultimo
    tratto, che il watcher dello stream controlla solo ogni 20 token), poi
    di nuovo le etichette: il taglio puo' lasciarne una da sola."""
    from llm_helpers import strip_leading_labels, truncate_prompt_echo
    text = strip_leading_labels(text or "", query)
    if echo_reference or echo_markers:
        text, _ = truncate_prompt_echo(text, echo_reference, markers=echo_markers)
    return strip_leading_labels(text, query).strip()


# Aggiunta in coda al prompt solo per il secondo tentativo. Rigenerare lo
# stesso prompt non basta: a temperatura 0.2 l'inizio e' quasi
# deterministico, e una richiesta che comincia ricopiando l'intestazione di
# un passaggio («[Cap. 5, §214 — …») lo rifa' identica (5 volte su 5). Con
# questa riga, 6 risposte vere su 6; alzando invece la temperatura a 0.6,
# 2 su 6 tornavano vuote.
_RETRY_NUDGE = ("\n\n(Rispondi con parole tue alla richiesta: non ricopiare "
                "i passaggi, gli esempi o le istruzioni qui sopra.)")
_NO_ANSWER = "The model did not produce an answer this time. Please ask again."




def _check_citations(resp, condition, route, chunks, query=""):
    """(testo, controllo) per la risposta finale. Enciclica: prima si
    correggono le citazioni sbagliate (cite_verify.repair_citations: numero
    di paragrafo corretto, parafrasi senza virgolette, frasi inventate tolte),
    poi si verifica quello che resta. Documenti: gli articoli citati devono
    essere fra i testi recuperati e le frasi fra virgolette devono starci
    parola per parola (docs_rag.check_documents). Senza framing non c'e'
    niente da controllare. Fail-open: se qualcosa va storto, testo intatto."""
    if condition == "baseline" or route == "unframed":
        return resp, None
    try:
        if route.startswith("documents"):
            import docs_rag
            return docs_rag.check_documents(resp, chunks, query)
        from cite_verify import verify_response, repair_citations
        resp, fixed = repair_citations(resp)
        report = verify_response(resp)
        report["repaired"] = fixed
        if route == "principles" and chunks:
            # Rifiuto in Documents: accanto all'enciclica ci sono i testi di
            # legge, e gli articoli citati si controllano su quelli.
            import docs_rag
            report["documents"] = docs_rag.check_documents(resp, chunks, query, quotes=False)[1]
        return resp, report
    except Exception:
        return resp, None


def _nothing_left(text):
    """Dopo la pulizia non resta una risposta: il modello ha ricopiato il
    prompt fin dalla prima parola. Si rigenera una volta."""
    return len(re.findall(r"\w", text or "")) < 20


def run_inference(condition: str, query: str, load_t: float,
                  max_tokens: int = 6000, rag_sources: list = None,
                  strict_ethics: bool = True, fewshot: bool = True,
                  self_critique: bool = False, structured: bool = False,
                  query_expansion: bool = True,
                  enable_thinking: bool = False,
                  ethical_gate: bool = True, documents: str = None) -> dict:
    """Esegue una singola inferenza su Apertus. Assume modello già caricato
    in ModelCache.

    enable_thinking: attiva la modalità "Deliberation" nativa di Apertus
    (sperimentale — vedi llm_helpers.format_apertus). Non testata con
    un'inferenza reale."""
    from llm_helpers import chat_complete
    nctx = getattr(ModelCache, "n_ctx", 8192)
    system, user_final, chunks, notice, MAX_TOK, echo_reference, echo_markers, route = _prepare_prompt(
        condition, query, rag_sources, strict_ethics, fewshot, structured,
        query_expansion, nctx, max_tokens, ethical_gate, documents)

    llm, _ = ModelCache.get(MODEL)  # riferimento locale, sicuro anche se scade nel frattempo
    t0 = time.time()
    r = chat_complete(llm, MODEL, system, user_final,
                      max_tokens=MAX_TOK, temperature=0.2,
                      repeat_penalty=1.1, top_p=0.95,
                      enable_thinking=enable_thinking,
                      echo_reference=echo_reference, echo_markers=echo_markers)
    resp = _clean_response(r["response"], query, echo_reference, echo_markers)
    tokens = r["tokens"]
    looped = r.get("repetition_truncated", False)
    echoed = r.get("prompt_echo_truncated", False)
    if _nothing_left(resp):
        print("  [regenerated: nothing left after cleanup]", flush=True)
        r = chat_complete(llm, MODEL, system, user_final + _RETRY_NUDGE,
                          max_tokens=MAX_TOK, temperature=0.2,
                          repeat_penalty=1.1, top_p=0.95,
                          enable_thinking=enable_thinking,
                          echo_reference=echo_reference, echo_markers=echo_markers)
        resp = _clean_response(r["response"], query, echo_reference, echo_markers)
        if _nothing_left(resp):
            resp = _NO_ANSWER
        tokens += r["tokens"]
        looped = looped or r.get("repetition_truncated", False)
        echoed = True
    inf_t = round(time.time()-t0, 1)

    # ── Passo 2 — Self-critique (Constitutional AI) ──────────────────────────
    draft = None
    if self_critique and condition != "baseline":
        draft = resp
        critique_user = CRITIQUE_TEMPLATE.format(query=query, draft=draft)
        r2 = chat_complete(llm, MODEL, system, critique_user,
                           max_tokens=MAX_TOK, temperature=0.2,
                           repeat_penalty=1.1, top_p=0.95,
                           enable_thinking=enable_thinking)
        inf_t = round(time.time()-t0, 1)  # tempo totale dei due passi
        resp = _strip_critique_preamble(r2["response"].strip()) or draft
        tokens = tokens + r2["tokens"]
        looped = looped or r2.get("repetition_truncated", False)
        r = r2  # per finish_reason

    # Ciclo o eco del prompt (vedi llm_helpers): il testo è già ripulito.
    # Non si dice all'utente — è un dettaglio del meccanismo, non della
    # risposta — ma resta nel log del server per chi fa diagnosi.
    _log_truncation(looped, echoed)

    tps = round(tokens / max(inf_t, 0.1), 1)
    finish = r["finish_reason"]
    truncated = (finish == "length") or (r["tokens"] >= MAX_TOK - 5)
    resp, citations = _check_citations(resp, condition, route, chunks, query)
    return {
        "model": MODEL,
        "condition": condition,
        "response": resp,
        "draft": draft,
        "citations": citations,
        "notice": notice,
        "unframed": route == "unframed",
        "route": route,
        "documents": documents if route.startswith("documents") else None,
        "load_t": load_t,
        "inference_t": inf_t,
        "tokens": tokens,
        "tps": tps,
        "truncated": truncated,
        "max_tokens": MAX_TOK,
        "rag_chunks": [
            {
                "source":     c.get("source", "encyclical"),
                "ref":        c.get("ref") or (f"§{c.get('paragraph')}" if c.get("paragraph") else ""),
                "title":      c.get("title", ""),
                "chapter":    c.get("chapter", ""),
                "section":    c.get("section", ""),
                "similarity": c.get("similarity"),
                "text":       c.get("text")[:400]+("…" if len(c.get("text",""))>400 else ""),
                "paragraph":  c.get("paragraph"),
                "via":        c.get("via"),
            }
            for c in chunks
        ],
    }


def _check_model_available():
    gguf = TEXT_MODELS.get(MODEL)
    if gguf is None or not gguf.exists():
        return (f"No model file for {MODEL_LABEL} in models/gguf/. "
                "Build one with tools/build_apertus15.sh 8B (or 70B), see the README.")
    return None


@app.route("/api/status")
def api_status():
    """Che cosa c'e' in memoria adesso (per la riga sotto la scelta del modello)."""
    loaded = ModelCache.name if ModelCache.llm is not None else None
    return jsonify({"loaded": loaded, "loaded_label": _model_label(loaded) if loaded else None,
                    "busy": _GEN_LOCK.locked(), "idle_unload_s": IDLE_UNLOAD_S})


@app.route("/api/unload", methods=["POST"])
def api_unload():
    """Scarica subito il modello dalla GPU, se nessuna risposta e' in corso."""
    if not _GEN_LOCK.acquire(blocking=False):
        return jsonify({"error": "An answer is being written"}), 409
    try:
        with ModelCache.lock:
            ModelCache._free_locked()
    finally:
        _GEN_LOCK.release()
    return api_status()


@app.route("/api/chat", methods=["POST"])
@_serialized
def api_chat():
    try:
        data = request.get_json()
        condition = data.get("condition","baseline")
        query = (data.get("query","") or "").strip()
        max_tokens = int(data.get("max_tokens", 6000))
        rag_sources = data.get("rag_sources") or None
        strict_ethics = bool(data.get("strict_ethics", True))
        fewshot = bool(data.get("fewshot", True))
        self_critique = bool(data.get("self_critique", False))
        structured = bool(data.get("structured", False))
        query_expansion = bool(data.get("query_expansion", False))
        enable_thinking = bool(data.get("enable_thinking", False))
        ethical_gate = bool(data.get("ethical_gate", True))
        if not query: return jsonify({"error":"Empty query"}), 400
        err = _check_model_available()
        if err: return jsonify({"error": err}), 400
        _, load_t = ModelCache.get(MODEL)
        result = run_inference(condition, query, load_t, max_tokens, rag_sources, strict_ethics, fewshot, self_critique, structured, query_expansion, enable_thinking, ethical_gate, documents=data.get("documents"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _with_memory(query, history):
    """(conversazione, domanda riscritta) per una domanda di seguito, oppure
    (query, None) se non c'e' storia. Vedi chat_memory."""
    import chat_memory
    if not chat_memory.clean_history(history):
        return query, None
    return (chat_memory.conversation(query, history),
            chat_memory.standalone(ModelCache.get(MODEL)[0], query, history))


def _stream_events(condition, query, max_tokens, rag_sources, strict_ethics,
                   fewshot, structured, query_expansion, enable_thinking,
                   ethical_gate, documents=None, history=None, search_query=None):
    """Una risposta in streaming, come coppie (evento, dati): 'notice',
    'token' man mano, poi 'done' con i metadati (citazioni, chunk RAG,
    tempi) o 'error'. La usano /api/chat_stream e /api/compare_stream."""
    try:
        if not query:
            yield "error", {"error": "Empty query"}; return
        err = _check_model_available()
        if err:
            yield "error", {"error": err}; return
        # Prima del primo token passano il caricamento del modello (se era
        # scarico), il cancello (col 70B ragiona prima di decidere: 10-25 s)
        # e la ricerca dei testi. Senza dire niente la pagina sembra ferma e
        # la risposta "non in streaming": si dice a che punto si e'.
        loading = _loading_status()
        if loading:
            yield "status", {"status": loading}
        llm, load_t = ModelCache.get(MODEL)
        nctx = getattr(ModelCache, "n_ctx", 8192)
        if history:
            yield "status", {"status": "Reading the conversation…"}
            asked = query
            query, search_query = _with_memory(query, history)
            if search_query and search_query.strip() != asked.strip():
                yield "understood", {"query": search_query}
        yield "status", {"status": "Checking the request…" if ethical_gate and condition != "baseline"
                         else "Preparing…"}
        system, user_final, chunks, notice, MAX_TOK, echo_reference, echo_markers, route = _prepare_prompt(
            condition, query, rag_sources, strict_ethics, fewshot,
            structured, query_expansion, nctx, max_tokens, ethical_gate,
            documents, search_query=search_query)
        if notice:
            yield "notice", {"notice": notice}
        yield "status", {"status": "Writing…"}

        from llm_helpers import (chat_stream, truncate_repetition,
                                 truncate_near_repetition, strip_leading_labels)
        t0 = time.time()
        tokens = 0
        for attempt in range(2):
            full, n_tok, finish, looped, echoed = [], 0, "", False, False
            # L'inizio resta in attesa finche' non e' chiaro che non e'
            # un'etichetta («[Risposta esplicita]») o un'eco del prompt: se
            # lo si mostrasse subito, lo si vedrebbe comparire e sparire.
            pending, flowing = "", False
            prompt_user = user_final if attempt == 0 else user_final + _RETRY_NUDGE
            for ev in chat_stream(llm, MODEL, system, prompt_user,
                                  max_tokens=MAX_TOK, temperature=0.2,
                                  repeat_penalty=1.1, top_p=0.95,
                                  enable_thinking=enable_thinking,
                                  echo_reference=echo_reference, echo_markers=echo_markers):
                if "token" not in ev:
                    n_tok = ev.get("tokens", 0); finish = ev.get("finish_reason", "")
                    looped = ev.get("repetition_truncated", False)
                    echoed = ev.get("prompt_echo_truncated", False)
                    continue
                full.append(ev["token"])
                if flowing:
                    yield "token", {"t": ev["token"]}
                    continue
                pending += ev["token"]
                head = strip_leading_labels(pending, query)
                settled = len(head) >= 60 or (head.strip() and "\n" in head) or len(pending) > 400
                if settled and not _nothing_left(
                        _clean_response(head, query, echo_reference, echo_markers)):
                    flowing = True
                    yield "token", {"t": head}
            tokens += n_tok
            resp = "".join(full)
            # chat_stream si ferma appena rileva il ciclo/l'eco, ma qualche
            # residuo può essere già stato accumulato prima che scattasse il
            # controllo (ogni 20 token) — ripulisce l'eventuale avanzo.
            if looped:
                resp, _ = truncate_repetition(resp)
                # In aggiunta: il watcher si ferma anche sul giro a vuoto, e
                # il taglio del blocco ripetuto puo' lasciarne uno in coda.
                resp, _ = truncate_near_repetition(resp)
            resp = _clean_response(resp, query, echo_reference, echo_markers)
            if _nothing_left(resp):
                if attempt == 0:
                    print("  [regenerated: nothing left after cleanup]", flush=True)
                    yield "reset", {}
                    continue
                resp = _NO_ANSWER
            break
        inf_t = round(time.time() - t0, 1)
        _log_truncation(looped, echoed)

        resp, citations = _check_citations(resp, condition, route, chunks, query)
        # Il testo RIPULITO va rimandato al browser: alla fine dello
        # streaming la pagina ridisegna la risposta con Object.assign(
        # {response: <token grezzi>}, meta), quindi se meta non porta
        # "response" resta a schermo il testo grezzo — e nessuna delle
        # ripuliture (eco del prompt, ripetizioni, giri a vuoto) arrivava
        # mai all'utente, pur valendo per il citation check.
        yield "done", {
            "response": resp,
            "model": MODEL, "condition": condition, "load_t": load_t,
            "understood": search_query,
            "inference_t": inf_t, "tokens": tokens,
            "tps": round(tokens / max(inf_t, 0.1), 1),
            "truncated": (finish == "length") or (n_tok >= MAX_TOK - 5),
            "max_tokens": MAX_TOK, "notice": notice, "citations": citations,
            "unframed": route == "unframed", "route": route,
            "documents": documents if route.startswith("documents") else None,
            "rag_chunks": [{
                "source": c.get("source", "encyclical"),
                "ref": c.get("ref") or (f"§{c.get('paragraph')}" if c.get("paragraph") else ""),
                "title": c.get("title", ""), "chapter": c.get("chapter", ""),
                "section": c.get("section", ""), "similarity": c.get("similarity"),
                "rerank_score": c.get("rerank_score"),
                "text": c.get("text")[:400] + ("…" if len(c.get("text", "")) > 400 else ""),
                "paragraph": c.get("paragraph"),
                "via": c.get("via"),
            } for c in chunks],
        }
    except Exception as e:
        yield "error", {"error": _friendly_error(e), "traceback": traceback.format_exc()}


_GATE_SEEN = {}


def _gate_decision(query):
    """DANNOSA, DELICATA o NORMALE secondo il cancello scelto (GATE). Il
    cancello a esempi conosce solo DANNOSA e NORMALE. Nel dubbio DANNOSA.

    L'ultima decisione si ricorda: il confronto interroga il cancello per
    sapere se la colonna con framing serve, e subito dopo _prepare_prompt lo
    interrogherebbe di nuovo sulla stessa richiesta (10-25 s col 70B). E'
    greedy, quindi la seconda risposta sarebbe la stessa. Gli errori non si
    ricordano."""
    seen = (MODEL, GATE, query)
    if seen in _GATE_SEEN:
        return _GATE_SEEN[seen]
    try:
        llm = ModelCache.get(MODEL)[0]
        if GATE == "policy":
            from harm_gate import classify_with_policy
            out = classify_with_policy(llm, query)[0]
        else:
            from harm_gate import is_harmful
            out = "DANNOSA" if is_harmful(llm, MODEL, query)[0] else "NORMALE"
    except Exception:
        return "DANNOSA"
    _GATE_SEEN.clear()
    _GATE_SEEN[seen] = out
    return out


def _judged_harmless(query, condition, ethical_gate):
    """Il cancello, interrogato da solo (vedi _prepare_prompt): True solo se
    la richiesta e' NORMALE, cioe' se la colonna con framing sarebbe identica
    alla baseline. Nel dubbio False: si applica il framing."""
    if not ethical_gate or condition == "baseline":
        return False
    return _gate_decision(query) == "NORMALE"


def _same_as_baseline(condition):
    """La colonna con framing di un confronto su una richiesta innocua: il
    cancello la manda senza framing, quindi il prompt sarebbe identico a
    quello della baseline. Generarla davvero costava 2 secondi e mostrava
    differenze dovute solo al campionamento (temperatura 0.2), che chi
    guarda attribuiva al framing. Non si genera: la pagina ci ricopia la
    risposta della baseline, segnata "same answer as baseline"."""
    return {"condition": condition, "same_as_baseline": True, "unframed": True,
            "response": "", "citations": None, "notice": None, "rag_chunks": [],
            "inference_t": 0.0, "tokens": 0, "tps": 0.0, "truncated": False}


def _framing_options(data):
    """Le leve del framing come le accetta l'API; la pagina manda solo i
    default (vedi il commento al posto delle vecchie opzioni avanzate)."""
    return dict(
        rag_sources=data.get("rag_sources") or None,
        strict_ethics=bool(data.get("strict_ethics", True)),
        fewshot=bool(data.get("fewshot", True)),
        structured=bool(data.get("structured", False)),
        query_expansion=bool(data.get("query_expansion", False)),
        enable_thinking=bool(data.get("enable_thinking", False)),
        ethical_gate=bool(data.get("ethical_gate", True)),
        documents=data.get("documents"),
    )


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.route("/api/chat_stream", methods=["POST"])
def api_chat_stream():
    """Streaming SSE per la modalità single (senza self-critique, che richiede
    la risposta completa). Invia 'token' man mano, poi un evento 'done' con i
    metadati (citazioni, notice, chunk RAG, tempi)."""
    data = request.get_json()
    condition = data.get("condition", "baseline")
    query = (data.get("query", "") or "").strip()
    max_tokens = int(data.get("max_tokens", 6000))
    opts = _framing_options(data)

    def events():
        for event, payload in _stream_events(condition, query, max_tokens,
                                             history=data.get("history"), **opts):
            yield _sse(event, payload)

    return Response(stream_with_context(_serialized_stream(data.get("model"), events)),
                    mimetype="text/event-stream", headers=_SSE_HEADERS)


@app.route("/api/compare_stream", methods=["POST"])
def api_compare_stream():
    """Confronto in streaming: prima la risposta con framing, poi la
    baseline, una dopo l'altra (il modello in VRAM e' uno solo). Ogni evento
    porta "side": "framed" | "baseline". La baseline e' la stessa di
    /api/compare: nessuna leva del framing, nemmeno il cancello. Se il
    cancello giudica innocua la richiesta, la colonna con framing non si
    genera (vedi _same_as_baseline)."""
    data = request.get_json()
    framed_condition = data.get("framed_condition", "manifest_explicit")
    query = (data.get("query", "") or "").strip()
    max_tokens = int(data.get("max_tokens", 6000))
    opts = _framing_options(data)

    def generate():
        nonlocal query
        if framed_condition == "baseline":
            yield _sse("error", {"side": "framed",
                                 "error": "Compare needs a non-baseline condition"})
            return
        # Memoria della chat: la conversazione si legge una volta, qui, e
        # vale per tutte e due le colonne.
        search_query = None
        if data.get("history"):
            loading = _loading_status()
            if loading:
                yield _sse("status", {"status": loading, "side": "framed"})
            yield _sse("status", {"status": "Reading the conversation…", "side": "framed"})
            asked = query
            query, search_query = _with_memory(query, data.get("history"))
            if search_query and search_query.strip() != asked.strip():
                yield _sse("understood", {"query": search_query, "side": "framed"})
        # Se il cancello lascia passare, la colonna con framing sarebbe una
        # seconda baseline. Altrimenti la strada (rifiuto o richiesta
        # delicata) la sceglie _prepare_prompt, che ritrova la decisione gia'
        # presa: col cancello spento qui una richiesta DELICATA finiva sulla
        # strada del rifiuto, e nel confronto veniva rifiutata.
        runs = [("baseline", "baseline",
                 dict(opts, ethical_gate=False, enable_thinking=False))]
        if framed_condition.startswith("docs_"):
            # Documents: da innocua la risposta viene dai documenti, non e'
            # una seconda baseline. La strada la sceglie il cancello dentro
            # _prepare_prompt, quindi qui non si interroga.
            runs.insert(0, ("framed", framed_condition, opts))
        else:
            loading = _loading_status()
            if loading:
                yield _sse("status", {"status": loading, "side": "framed"})
                ModelCache.get(MODEL)
            yield _sse("status", {"status": "Checking the request…", "side": "framed"})
            if _judged_harmless(query, framed_condition, opts["ethical_gate"]):
                yield _sse("done", dict(_same_as_baseline(framed_condition), side="framed"))
            else:
                runs.insert(0, ("framed", framed_condition, opts))
        for side, condition, o in runs:
            for event, payload in _stream_events(condition, query, max_tokens,
                                                 search_query=search_query, **o):
                yield _sse(event, dict(payload, side=side))

    return Response(stream_with_context(
                        _serialized_stream(data.get("model"), generate, {"side": "framed"})),
                    mimetype="text/event-stream", headers=_SSE_HEADERS)


@app.route("/api/compare", methods=["POST"])
@_serialized
def api_compare():
    """Esegue 2 inferenze sulla stessa domanda: baseline + framed_condition.
    Stesso modello caricato → 2 forward passes consecutive, costo ~doppio dell'inferenza."""
    try:
        data = request.get_json()
        framed_condition = data.get("framed_condition","manifest_explicit")
        query = (data.get("query","") or "").strip()
        max_tokens = int(data.get("max_tokens", 6000))
        rag_sources = data.get("rag_sources") or None
        strict_ethics = bool(data.get("strict_ethics", True))
        fewshot = bool(data.get("fewshot", True))
        self_critique = bool(data.get("self_critique", False))
        structured = bool(data.get("structured", False))
        query_expansion = bool(data.get("query_expansion", False))
        enable_thinking = bool(data.get("enable_thinking", False))
        if not query: return jsonify({"error":"Empty query"}), 400
        if framed_condition == "baseline":
            return jsonify({"error":"Compare needs a non-baseline condition"}), 400
        err = _check_model_available()
        if err: return jsonify({"error": err}), 400
        # Carica modello una volta sola
        _, load_t = ModelCache.get(MODEL)
        # 1) baseline (mai strict, mai few-shot, mai critique, mai deliberation:
        #    riferimento "non guidato" allo stato più semplice possibile)
        base = run_inference("baseline", query, load_t, max_tokens)
        # 2) framed (load_t=0 perché modello già in VRAM)
        if framed_condition.startswith("docs_"):
            framed = run_inference(framed_condition, query, 0.0, max_tokens, rag_sources, strict_ethics, fewshot, self_critique, structured, query_expansion, enable_thinking, documents=data.get("documents"))
        elif _judged_harmless(query, framed_condition, bool(data.get("ethical_gate", True))):
            framed = dict(base, condition=framed_condition, same_as_baseline=True,
                          unframed=True, citations=None, notice=None)
        else:
            framed = run_inference(framed_condition, query, 0.0, max_tokens, rag_sources, strict_ethics, fewshot, self_critique, structured, query_expansion, enable_thinking,
                                   ethical_gate=bool(data.get("ethical_gate", True)))
        return jsonify({
            "model": MODEL,
            "load_t": load_t,
            "baseline": base,
            "framed":   framed,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    from model_cache import IDLE_UNLOAD_S
    print("="*60)
    print("  Apertus Playground (Magnifica Humanitas, RAG multi-fonte)")
    print("  URL: http://localhost:5002")
    print(f"  Idle unload: dopo {IDLE_UNLOAD_S:.0f}s senza richieste "
          f"(override con LLM_IDLE_UNLOAD_S)")
    print("="*60)
    start_idle_unload_thread()
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=False)
