#!/usr/bin/env python3
"""
Helper condivisi per inferenza LLM corretta:
  - n_ctx dinamico in base al context length nativo del modello
  - chat_complete(): usa il chat template NATIVO del GGUF via create_chat_completion,
    con fallback a 3 livelli:
      1) messages=[system, user]                     (modelli con system role)
      2) messages=[user] con system fuso nel prompt  (Gemma e simili: no system role)
      3) raw completion con formato generico         (leo/maestrale: no chat_template)

Questo sostituisce il vecchio approccio "raw completion con [SYSTEM]/[USER]" che
non rispettava il formato di instruction-tuning dei modelli.
"""
import time

# Context length nativo (da scan metadata GGUF). Usato per n_ctx dinamico.
MODEL_CTX_TRAIN = {
    "minerva:7b": 4096, "eurollm:9b": 4096, "lucie:7b": 4096, "yi15:9b": 4096,
    "llamantino:8b": 8192, "salamandra:7b": 8192, "gemma2:9b": 8192, "aya:8b": 8192,
    "mistral:7b": 32768, "leo:7b": 32768, "maestrale:7b": 32768, "occiglot:7b": 32768,
    "llama31:8b": 131072, "phi35:mini": 131072, "granite31:8b": 131072,
    "deepseek-r1:8b": 131072, "glm4:9b": 131072,
    "apertus:8b": 65536, "apertus15:8b": 65536, "apertus:70b": 65536,
    "apertus15:70b": 65536,
    "qwen25:7b": 32768,
}

# Modelli SENZA chat_template nel GGUF → richiedono raw completion
MODELS_NO_TEMPLATE = {"leo:7b", "maestrale:7b"}

# Cap massimo di n_ctx per non saturare la VRAM (RTX 4090 24GB).
# 16384 è ampiamente sufficiente per manifesto (2.3K) + RAG (3K) + risposta lunga.
N_CTX_CAP = 16384


def get_n_ctx(model_key: str) -> int:
    """Restituisce il n_ctx ottimale: min(ctx_train, cap), ma almeno 4096."""
    train = MODEL_CTX_TRAIN.get(model_key, 8192)
    return max(4096, min(train, N_CTX_CAP))


def format_raw(system: str, user: str) -> str:
    """Formato generico per modelli senza chat_template (leo/maestrale)."""
    if system:
        return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
    return f"[USER]\n{user}\n\n[ASSISTANT]\n"


RAW_STOP = ["[USER]", "[SYSTEM]", "[ASSISTANT]",
            "<|im_end|>", "</s>", "<|user|>", "<|system|>", "<|assistant|>"]


# ── Apertus: modalità "Deliberation" (reasoning interno) ─────────────────────
# Il chat template incorporato nel GGUF di Apertus (SwissAI) supporta un flag
# enable_thinking che, se attivo, dichiara "Deliberation: enabled" nel turno
# "developer" — spento di default. Verificato leggendo a mano i metadata del
# file GGUF (byte del blob chat_template): create_chat_completion() di
# llama-cpp-python 0.3.28 non espone un modo per passare variabili extra al
# render jinja del template, quindi qui ricostruiamo il prompt letteralmente
# come lo produrrebbe il template, usando i suoi stessi token speciali.
# Testato dal vivo (23/09/2026) su una batteria di 12 casi: acceso porta le
# richieste dannose correttamente rifiutate da 6/7 a 7/7, compresa quella che
# altrimenti passava senza che nessun controllo se ne accorgesse. Ora e' acceso
# di default. Il modello non emette un blocco di ragionamento visibile: cambia
# il turno developer, e con esso il comportamento.
APERTUS_MODEL_KEY = "apertus:8b"


def is_apertus(model_key: str) -> bool:
    """Tutte le versioni di Apertus hanno lo stesso formato di chat (token
    61-68, turno developer con la Deliberation): verificato che il template
    di v1.5 e quello di 2509 producono lo stesso prompt di format_apertus."""
    return (model_key or "").startswith("apertus")
_AP_SYSTEM_START, _AP_SYSTEM_END = "<|system_start|>", "<|system_end|>"
_AP_DEV_START, _AP_DEV_END = "<|developer_start|>", "<|developer_end|>"
_AP_USER_START, _AP_USER_END = "<|user_start|>", "<|user_end|>"
_AP_ASSISTANT_START = "<|assistant_start|>"


def format_apertus(system: str, user: str, enable_thinking: bool = True) -> str:
    """Prompt Apertus costruito a mano coi token del suo chat template, con la
    Deliberation esplicitamente accesa o spenta nel turno developer."""
    sys_text = system or (
        "You are Apertus, a helpful assistant created by the SwissAI initiative."
    )
    deliberation = "enabled" if enable_thinking else "disabled"
    return (
        f"{_AP_SYSTEM_START}{sys_text}{_AP_SYSTEM_END}"
        f"{_AP_DEV_START}Deliberation: {deliberation}\nTool Capabilities: disabled{_AP_DEV_END}"
        f"{_AP_USER_START}{user}{_AP_USER_END}"
        f"{_AP_ASSISTANT_START}"
    )


def truncate_repetition(text: str, min_block_len: int = 50, min_repeats: int = 3):
    """Rileva un ciclo degenerato: lo stesso blocco di testo (separato da riga
    vuota, almeno min_block_len caratteri) ripetuto min_repeats+ volte di
    fila — osservato dal vivo: il modello non si fermava da solo e saturava
    max_tokens ripetendo la stessa risposta (6000 token, 76s sprecati).
    Se lo trova, tronca subito dopo la PRIMA occorrenza (tiene una sola
    copia). Ritorna (testo_pulito, è_stato_troncato)."""
    blocks = [b.strip() for b in text.split("\n\n")]
    counts = {}
    for b in blocks:
        if len(b) >= min_block_len:
            counts[b] = counts.get(b, 0) + 1
    for key, n in counts.items():
        if n >= min_repeats:
            idx = text.find(key)
            if idx == -1:
                continue
            return text[:idx + len(key)].strip(), True
    return text, False


_NEAR_STOP = {
    "il","lo","la","i","gli","le","un","uno","una","del","dello","della","dei",
    "degli","delle","di","a","da","in","con","su","per","tra","fra","al","allo",
    "alla","ai","agli","alle","dal","dalla","nel","nella","nei","sul","sulla",
    "non","che","chi","cui","e","ed","o","od","ma","se","come","anche","più",
    "essere","viene","sono","ha","ho","hai","si","ci","questo","questa","quello",
    "quella","suo","sua","loro","ogni","tutto","tutti","modo","puo","può",
}


def _content_stems(par: str) -> set:
    """Parole-contenuto di un paragrafo, ridotte allo stem (primi 5 caratteri):
    robusto a flessioni e riformulazioni leggere."""
    import re as _re
    words = _re.sub(r"[^a-zàèéìòù0-9 ]+", " ", par.lower()).split()
    return {w[:5] for w in words if len(w) >= 4 and w not in _NEAR_STOP}


def truncate_near_repetition(text: str, min_par_len: int = 110,
                             min_novelty: float = 0.15):
    """Rileva la ripetizione PARAFRASATA: un paragrafo che ridice, con parole
    un po' diverse, quello che un paragrafo precedente ha già detto.

    truncate_repetition prende solo i blocchi identici ripetuti 3+ volte, ma
    dal vivo il guasto più frequente è un altro: il modello dà la risposta
    giusta e poi continua a girarci intorno. Esempio reale, sul caso del
    padre anziano — «Invece, potresti aiutarlo a gestire i conti in modo più
    sicuro» e due paragrafi dopo «In questo modo, potresti aiutarlo a gestire
    i conti in modo più sicuro»: nessuna ripetizione esatta, nessun blocco
    triplicato, ma tre quarti dei token generati sono divagazione — ed è lì
    che spuntano le citazioni attribuite a paragrafi a caso.

    Il criterio è la NOVITÀ, non la somiglianza: quante parole-contenuto del
    paragrafo non erano già comparse prima. Misurare invece la sovrapposizione
    col testo precedente non funziona — provato: un'alternativa costruttiva
    ("invece posso scriverti un post che spieghi come rendere i servizi più
    accessibili") riusa per forza il lessico del tema, e veniva tagliata
    insieme alla divagazione vera.

    Soglia scelta sui valori misurati su risposte reali: i paragrafi da
    buttare stanno a 0.08 e a 0.00 (su una risposta il modello ha ripetuto
    *identici* un paragrafo e un elenco, e truncate_repetition non li prende
    perché ne pretende tre), mentre il contenuto legittimo più povero di
    novità — un esempio citato a blocco — sta a 0.21. Fra 0.08 e 0.21 c'è
    spazio: tagliamo solo ciò che non aggiunge quasi nulla, lasciando stare
    il resto, anche a costo di sopportare qualche paragrafo prolisso.
    Ritorna (testo_pulito, è_stato_troncato)."""
    import re as _re
    # Un elenco riprende per forza il lessico della frase che lo introduce:
    # misurarne la sovrapposizione non dice niente. Su una risposta reale il
    # taglio aveva portato via l'elenco dei suggerimenti concreti lasciando
    # appeso un "suggerisco di:" — il contenuto utile, non la divagazione.
    is_list = _re.compile(r"^\s*(?:[-*•]|\d+[.)])\s").match
    paras = text.split("\n\n")
    visti, keep = set(), []
    for p in paras:
        body = p.strip()
        st = _content_stems(body)
        if len(body) >= min_par_len and st and visti:
            if len(st - visti) / len(st) < min_novelty:
                # Non tagliare se quello che resta finisce con un'introduzione
                # sospesa ("Per questo motivo, suggerisco di:"): il paragrafo
                # che segue la completa, non la ripete.
                tenuto = "\n\n".join(keep).rstrip()
                if not tenuto.endswith(":"):
                    return tenuto.strip(), True
        visti |= st
        keep.append(p)
    return text, False


import re as _re_labels

# Etichetta messa dal modello davanti alla risposta vera, imitando il formato
# del prompt: «[Risposta esplicita]», «[Risposta dell'Assistente AI]»,
# «[Esempio di risposta a una richiesta delicata ma legittima]», «Risposta:».
# Contate 44 su 616 risposte salvate. Con Apertus 70B anche «Richiesta
# dell'utente: …» / «Richiesta reale:» seguite dalla domanda ricopiata (la
# domanda si toglie qui sotto, se identica). Fra parentesi quadre solo se contiene
# una di queste parole: un modello di lettera che apre con «[Nome]» resta com'e'.
_LEADING_LABEL = _re_labels.compile(
    r"\A\s*(?:\*\*)?"
    r"(?:\[[^\]\n]{0,100}\b(?:risposta|esempio|answer|response|assistente|assistant)\b[^\]\n]{0,100}\]"
    r"|(?:risposta(?:\s+(?:corretta|esplicita|finale))?|answer|response"
    r"|richiesta(?:\s+(?:reale|dell'utente))?)\s*:)"
    r"(?:\*\*)?[ \t]*(?:\n|(?=\S))",
    _re_labels.IGNORECASE)


def _plain(s: str) -> str:
    return " ".join(_re_labels.sub(r"[^\w\s]", " ", s.lower()).split())


def strip_leading_labels(text: str, query: str = None) -> str:
    """Toglie le etichette iniziali (vedi _LEADING_LABEL), anche piu' d'una
    di fila, e un primo paragrafo che ripete la richiesta parola per parola
    (osservato dopo «[Risposta dell'Assistente AI]»)."""
    if not text:
        return text
    while True:
        m = _LEADING_LABEL.match(text)
        if not m or not m.end():
            break
        text = text[m.end():]
    text = text.lstrip()
    if query:
        first, sep, rest = text.partition("\n\n")
        if sep and _plain(first) == _plain(query):
            text = rest.lstrip()
    return text


def truncate_prompt_echo(text: str, reference: str, min_match_len: int = 45,
                         markers: list = None):
    """Rileva quando il modello smette di rispondere e comincia a ricopiare
    testo letterale dal PROPRIO PROMPT (es. le istruzioni few-shot) invece
    di continuare a generare una risposta fresca — diverso da
    truncate_repetition, che rileva un blocco ripetuto DENTRO la risposta:
    qui il testo copiato non è ripetuto, compare una sola volta, ma
    proviene dal prompt (osservato dal vivo su una lettera di
    licenziamento: dopo una risposta corretta e specifica il modello ha
    continuato a generare oltre la risposta, ricopiando parola per parola
    il footer delle istruzioni few-shot — "la tua risposta deve essere
    specifica... non hai risposto alla richiesta reale" — e poi l'esempio A
    stesso, etichettato "Risposta corretta:«...»" — testo mai inteso per
    l'utente finale).

    reference deve essere SOLO il testo istruzionale iniettato nel prompt
    (es. _FEWSHOT_HEADER + _FEWSHOT_FOOTER), MAI il corpo degli esempi: la
    risposta genuina imita legittimamente lo STILE e spesso riusa FRASI
    intere degli esempi (è lo scopo del few-shot) — includere i loro corpi
    qui farebbe scattare falsi positivi enormi, troncando risposte corrette
    fin dalla prima frase (verificato: con gli esempi nel reference, una
    risposta genuina che apre come l'esempio A veniva svuotata del tutto).
    Le citazioni letterali dei passaggi RAG sono volute allo stesso modo,
    quindi non vanno bloccate: reference non deve includerle.

    markers: stringhe BREVI e inequivocabili (es. "Risposta corretta:", il
    titolo di un esempio) che non compaiono MAI in una risposta genuina,
    indipendentemente da quanto sia lunga la corrispondenza — osservato dal
    vivo su una richiesta banale (invitare un collega a pranzo): il modello
    ha rielaborato ENTRAMBI gli esempi come se fossero la risposta,
    cambiando persino il loro involucro ("[Esempio A — ...]" diventato
    "**Esempio A — ...**" in grassetto) pur mantenendo intatta l'etichetta
    testuale. Una finestra scorrevole sul solo reference non li cattura
    (il corpo degli esempi non c'è, di proposito — vedi sopra); questi
    marker coprono quel buco senza doverli allungare abbastanza da rischiare
    falsi positivi.

    Cerca la prima finestra di reference lunga min_match_len caratteri o il
    primo marker che compare in text, tronca al primo dei due (quello più
    vicino all'inizio) e rifila all'ultimo confine di frase per non
    lasciare un frammento a metà. Ritorna (testo_pulito, troncato)."""
    if not text:
        return text, False
    earliest = None
    if reference and len(reference) >= min_match_len:
        windows = {reference[i:i + min_match_len]
                   for i in range(len(reference) - min_match_len + 1)}
        for i in range(len(text) - min_match_len + 1):
            if text[i:i + min_match_len] in windows:
                earliest = i
                break
    for m in (markers or []):
        idx = text.find(m)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text, False
    cut = text[:earliest].rstrip()
    # Se prima dell'eco non c'era nessuna parola vera, non lasciare in giro
    # il resto dell'involucro: un «[» o un «**» orfano da solo in risposta.
    import re as _re
    if not _re.search(r"\w", cut):
        return "", True
    # Rifila all'ultimo confine di frase per non lasciare un frammento a
    # metà (es. "...deve essere specifica a queste circostanze citate —
    # nominando"). Cerca all'indietro fino a 200 caratteri; oltre, meglio
    # tenere il taglio grezzo che perdere troppo testo genuino.
    boundary = max(cut.rfind(s) for s in (". ", "! ", "? ", ".\n", "!\n", "?\n"))
    if boundary != -1 and len(cut) - boundary < 200:
        cut = cut[:boundary + 1].rstrip()
    return cut, True


def _reset(llm):
    """ModelCache riusa lo stesso oggetto Llama fra domande diverse (per non
    ricaricare il modello ogni volta). Senza reset esplicito la KV-cache
    interna di llama-cpp-python può non svuotarsi del tutto fra una
    generazione e l'altra, e il modello 'vede' ancora frammenti del prompt
    precedente. Va chiamato prima di OGNI nuova generazione indipendente."""
    try:
        llm.reset()
    except Exception:
        pass


def chat_complete(llm, model_key: str, system: str, user: str,
                  max_tokens: int = 1500, temperature: float = 0.2,
                  repeat_penalty: float = 1.1, top_p: float = 0.95,
                  enable_thinking: bool = False,
                  echo_reference: str = None, echo_markers: list = None) -> dict:
    """
    Esegue una completion usando il chat template nativo quando disponibile.
    Ritorna dict uniforme: {response, tokens, finish_reason, method,
    repetition_truncated, prompt_echo_truncated}.

    `method` indica quale percorso è stato usato (per debug/telemetria):
      'chat_system' | 'chat_merged' | 'raw' | 'apertus_deliberation'

    enable_thinking: solo per model_key == 'apertus:8b' (vedi format_apertus).
    Ignorato per tutti gli altri modelli.

    echo_reference / echo_markers: vedi truncate_prompt_echo — taglia la
    risposta se comincia a ricopiare testo letterale dal prompt invece di
    rispondere.
    """
    result = _chat_complete_raw(llm, model_key, system, user, max_tokens,
                                temperature, repeat_penalty, top_p, enable_thinking)
    cleaned, truncated = truncate_repetition(result["response"])
    # In aggiunta, non in alternativa: il taglio del blocco ripetuto puo'
    # lasciare in coda un giro a vuoto che non e' una ripetizione esatta.
    cleaned, girato = truncate_near_repetition(cleaned)
    truncated = truncated or girato
    echoed = False
    if echo_reference or echo_markers:
        cleaned, echoed = truncate_prompt_echo(cleaned, echo_reference, markers=echo_markers)
    result["response"] = cleaned
    result["repetition_truncated"] = truncated
    result["prompt_echo_truncated"] = echoed
    return result


def _chat_complete_raw(llm, model_key: str, system: str, user: str,
                       max_tokens: int, temperature: float,
                       repeat_penalty: float, top_p: float,
                       enable_thinking: bool) -> dict:
    _reset(llm)
    # ── Caso 0: Apertus con Deliberation esplicitamente richiesta ────────────
    if is_apertus(model_key) and enable_thinking:
        prompt = format_apertus(system, user, enable_thinking=True)
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature,
                  repeat_penalty=repeat_penalty, top_p=top_p, echo=False)
        ch = out["choices"][0]
        return {
            "response": ch["text"].strip(),
            "tokens": out["usage"]["completion_tokens"],
            "finish_reason": ch.get("finish_reason", ""),
            "method": "apertus_deliberation",
        }
    # ── Caso 3: modello senza chat_template → raw ────────────────────────────
    if model_key in MODELS_NO_TEMPLATE:
        prompt = format_raw(system, user)
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature,
                  repeat_penalty=repeat_penalty, top_p=top_p, echo=False,
                  stop=RAW_STOP)
        ch = out["choices"][0]
        return {
            "response": ch["text"].strip(),
            "tokens": out["usage"]["completion_tokens"],
            "finish_reason": ch.get("finish_reason", ""),
            "method": "raw",
        }

    # ── Caso 1: chat con ruolo system ────────────────────────────────────────
    if system:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
    else:
        messages = [{"role": "user", "content": user}]
    try:
        out = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, repeat_penalty=repeat_penalty, top_p=top_p)
        ch = out["choices"][0]
        return {
            "response": ch["message"]["content"].strip(),
            "tokens": out["usage"]["completion_tokens"],
            "finish_reason": ch.get("finish_reason", ""),
            "method": "chat_system" if system else "chat_user",
        }
    except Exception as e:
        # ── Caso 2: il modello rifiuta il system role (Gemma) → fondi in user ─
        if system and ("system" in str(e).lower() or "role" in str(e).lower()):
            merged_user = (
                f"Istruzioni di sistema (da seguire sempre):\n{system}\n\n"
                f"---\n\nRichiesta dell'utente:\n{user}"
            )
            _reset(llm)
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": merged_user}],
                max_tokens=max_tokens, temperature=temperature,
                repeat_penalty=repeat_penalty, top_p=top_p)
            ch = out["choices"][0]
            return {
                "response": ch["message"]["content"].strip(),
                "tokens": out["usage"]["completion_tokens"],
                "finish_reason": ch.get("finish_reason", ""),
                "method": "chat_merged",
            }
        # Altro errore: rilancia
        raise


class _RepetitionWatcher:
    """Osserva i token man mano che arrivano e segnala un ciclo degenerato
    (stesso blocco ripetuto, vedi truncate_repetition) senza ricontrollare
    ad ogni singolo token — solo ogni check_every, per non appesantire lo
    streaming. Usato per FERMARE la generazione in anticipo invece di
    scoprire il ciclo solo a fine risposta (osservato dal vivo: 6000 token,
    76s, tutti sprecati in un paragrafo ripetuto ~30 volte)."""
    def __init__(self, check_every: int = 20):
        self.buf = []
        self.n = 0
        self.check_every = check_every

    def add(self, token: str) -> bool:
        self.buf.append(token)
        self.n += 1
        if self.n % self.check_every:
            return False
        text = "".join(self.buf)
        _, truncated = truncate_repetition(text)
        if truncated:
            return True
        # Anche il giro a vuoto (paragrafo che non aggiunge quasi nulla) vale
        # come stop: e' il guasto piu' frequente e spreca fino a tre quarti
        # dei token. Si guarda solo la parte gia' chiusa da una riga vuota,
        # perche' l'ultimo paragrafo e' ancora a meta' e sembrerebbe povero
        # di novita' solo perche' incompleto.
        done = text.rpartition("\n\n")[0]
        if not done:
            return False
        _, girato = truncate_near_repetition(done)
        return girato


class _PromptEchoWatcher:
    """Come _RepetitionWatcher ma per l'eco dal prompt (vedi
    truncate_prompt_echo): individua quando lo stream inizia a ricopiare
    testo letterale iniettato nel prompt (es. gli esempi few-shot) invece
    di rispondere, e permette di fermare la generazione in anticipo invece
    di sprecare token fino a max_tokens. Senza reference né markers, add()
    non fa mai scattare nulla (nessun costo per i chiamanti che non li
    passano)."""
    def __init__(self, reference: str = None, markers: list = None, check_every: int = 20):
        self.reference = reference or ""
        self.markers = markers or []
        self.buf = []
        self.n = 0
        self.check_every = check_every

    def add(self, token: str) -> bool:
        if not self.reference and not self.markers:
            return False
        self.buf.append(token)
        self.n += 1
        if self.n % self.check_every:
            return False
        _, echoed = truncate_prompt_echo("".join(self.buf), self.reference, markers=self.markers)
        return echoed


def chat_stream(llm, model_key: str, system: str, user: str,
                max_tokens: int = 1500, temperature: float = 0.2,
                repeat_penalty: float = 1.1, top_p: float = 0.95,
                enable_thinking: bool = False, echo_reference: str = None,
                echo_markers: list = None):
    """Versione streaming di chat_complete: generatore che yield-a dict.

      {"token": "..."}   per ogni pezzo di testo prodotto
      {"done": True, "tokens": n, "finish_reason": ..., "method": ...,
       "repetition_truncated": bool, "prompt_echo_truncated": bool}  alla fine

    Rispetta la stessa logica (apertus_deliberation / raw / chat_system / chat_merged).
    Si ferma da solo se rileva un ciclo degenerato (vedi _RepetitionWatcher) o
    un'eco del prompt (vedi _PromptEchoWatcher, attivo solo se echo_reference/
    echo_markers sono dati), invece di continuare fino a max_tokens.
    """
    import itertools

    _reset(llm)
    # ── Caso 0: Apertus con Deliberation esplicitamente richiesta ────────────
    if is_apertus(model_key) and enable_thinking:
        prompt = format_apertus(system, user, enable_thinking=True)
        stream = llm(prompt, max_tokens=max_tokens, temperature=temperature,
                     repeat_penalty=repeat_penalty, top_p=top_p, echo=False,
                     stream=True)
        n, finish = 0, ""
        watcher = _RepetitionWatcher()
        echo_watcher = _PromptEchoWatcher(echo_reference, echo_markers)
        looped = echoed = False
        for ch in stream:
            c = ch["choices"][0]
            t = c.get("text", "") or ""
            if t:
                n += 1; yield {"token": t}
                if watcher.add(t):
                    looped = True
                    break
                if echo_watcher.add(t):
                    echoed = True
                    break
            if c.get("finish_reason"):
                finish = c["finish_reason"]
        yield {"done": True, "tokens": n, "finish_reason": finish,
              "method": "apertus_deliberation", "repetition_truncated": looped,
              "prompt_echo_truncated": echoed}
        return

    # ── Caso 3: modello senza chat_template → raw ────────────────────────────
    if model_key in MODELS_NO_TEMPLATE:
        prompt = format_raw(system, user)
        stream = llm(prompt, max_tokens=max_tokens, temperature=temperature,
                     repeat_penalty=repeat_penalty, top_p=top_p, echo=False,
                     stop=RAW_STOP, stream=True)
        n, finish = 0, ""
        watcher = _RepetitionWatcher()
        echo_watcher = _PromptEchoWatcher(echo_reference, echo_markers)
        looped = echoed = False
        for ch in stream:
            c = ch["choices"][0]
            t = c.get("text", "") or ""
            if t:
                n += 1; yield {"token": t}
                if watcher.add(t):
                    looped = True
                    break
                if echo_watcher.add(t):
                    echoed = True
                    break
            if c.get("finish_reason"):
                finish = c["finish_reason"]
        yield {"done": True, "tokens": n, "finish_reason": finish,
              "method": "raw", "repetition_truncated": looped,
              "prompt_echo_truncated": echoed}
        return

    # ── Caso 1/2: chat template nativo, con fallback merged per Gemma ─────────
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
    method = "chat_system" if system else "chat_user"

    def _make(msgs):
        return llm.create_chat_completion(
            messages=msgs, max_tokens=max_tokens, temperature=temperature,
            repeat_penalty=repeat_penalty, top_p=top_p, stream=True)

    # L'errore "system role non supportato" può emergere alla prima iterazione:
    # proviamo a estrarre il primo chunk dentro try, e in caso ripieghiamo.
    try:
        gen = _make(messages)
        first = next(gen)
    except StopIteration:
        first, gen = None, iter(())
    except Exception as e:
        if system and ("system" in str(e).lower() or "role" in str(e).lower()):
            merged_user = (
                f"Istruzioni di sistema (da seguire sempre):\n{system}\n\n"
                f"---\n\nRichiesta dell'utente:\n{user}"
            )
            method = "chat_merged"
            _reset(llm)
            gen = _make([{"role": "user", "content": merged_user}])
            try:
                first = next(gen)
            except StopIteration:
                first, gen = None, iter(())
        else:
            raise

    n, finish = 0, ""
    watcher = _RepetitionWatcher()
    echo_watcher = _PromptEchoWatcher(echo_reference, echo_markers)
    looped = echoed = False
    for ch in itertools.chain([first] if first is not None else [], gen):
        c = ch["choices"][0]
        t = (c.get("delta") or {}).get("content", "") or ""
        if t:
            n += 1; yield {"token": t}
            if watcher.add(t):
                looped = True
                break
            if echo_watcher.add(t):
                echoed = True
                break
        if c.get("finish_reason"):
            finish = c["finish_reason"]
    yield {"done": True, "tokens": n, "finish_reason": finish, "method": method,
          "repetition_truncated": looped, "prompt_echo_truncated": echoed}
