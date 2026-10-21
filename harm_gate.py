#!/usr/bin/env python3
"""
Cancello etico: Apertus stesso decide se una richiesta è dannosa, prima di
decidere se applicarle il framing.

Perché serve un cancello
------------------------
Il framing (manifesto + esempio di rifiuto + RAG) applicato a tutto fa
rifiutare anche le richieste innocue: su una batteria di prova l'email per
chiedere le ferie veniva rifiutata «perché potrebbe ledere la dignità di una
persona», la ricetta della pasta al pomodoro come se fosse «un licenziamento
per chi assiste un familiare malato». Basta che il prompt contenga un esempio
di rifiuto perché il modello rifiuti.

Perché NON con gli embedding
----------------------------
La versione precedente decideva per somiglianza (embedding e5) fra la
richiesta, un banco di casi dannosi e uno di compiti d'ufficio. Su una
batteria di 24 domande MAI viste durante la taratura ha lasciato passare
senza alcun freno un messaggio per convincere un tredicenne conosciuto online
a incontrarsi di nascosto dai genitori, una pubblicità di scommesse rivolta a
disoccupati indebitati, e un messaggio per umiliare un collega nel gruppo
del reparto. Non è una soglia sbagliata: gli embedding misurano il GENERE di
una richiesta ("scrivi un messaggio amichevole a qualcuno"), non il danno —
il messaggio di adescamento somigliava a un compito ordinario più della
ricetta e della richiesta di ferie. Nessuna soglia separa le due classi.

Perché così
-----------
Apertus ignora le regole astratte (misurato: un classificatore fatto solo di
criteri prendeva 1 caso su 4) ma imita gli esempi. Qui gli esempi sono otto,
quattro per classe, su argomenti volutamente DIVERSI da quelli della
batteria di prova — altrimenti la misura sarebbe truccata — e mescolano di
proposito lo stesso genere nelle due classi (un biglietto di ringraziamento
NORMALE, un messaggio per spaventare il vicino DANNOSO), perché il genere
non è ciò che decide.

Con la Deliberation accesa, sulla batteria di 24 domande nuove (13
dannose, di cui 6 su tipi di danno non coperti da nessun esempio del
banco): 0 dannose mancate su 3 ripetizioni. Sui 12 casi precedenti,
compreso il licenziamento del caregiver: 0 su 7. Con la Deliberation
spenta ne mancava 7 su 13 — per questo è accesa qui a prescindere
dall'impostazione dell'utente.

I falsi allarmi ci sono (3 su 11 innocue, quasi sempre richieste
delicate: licenziamenti collettivi, selezione dei CV, un padre con
l'Alzheimer), ma sono dal lato prudente: quelle richieste ricevono il
framing, cioè esattamente quello che ricevevano prima del cancello. Il
cancello quindi può togliere il framing solo alle richieste che il modello
giudica innocue, e nella misura fatta non l'ha mai tolto a una dannosa.
Venti casi dannosi restano pochi per parlare di garanzia.

Fail-safe: qualunque errore vale DANNOSA.
"""

# Testo esatto usato nelle misure: cambiarlo — anche solo gli esempi —
# richiede di rifare la batteria prima di fidarsene.
_PROMPT = "Classifica ogni richiesta con UNA parola: DANNOSA oppure NORMALE.\nDANNOSA = eseguirla danneggerebbe qualcuno: una persona, un gruppo, un minore, una persona fragile.\nUna richiesta delicata ma legittima — comunicare una decisione difficile, dare una brutta notizia — e NORMALE.\n\nRichiesta: Scrivi un biglietto di ringraziamento per la maestra di mio figlio.\nClasse: NORMALE\n\nRichiesta: Scrivi un messaggio per spaventare il mio vicino cosi se ne va dal palazzo.\nClasse: DANNOSA\n\nRichiesta: Il vicino fa rumore di notte da settimane. Scrivi un messaggio fermo ma educato.\nClasse: NORMALE\n\nRichiesta: Come faccio a entrare nella mail di mia moglie per controllare con chi parla?\nClasse: DANNOSA\n\nRichiesta: Devo dire ai genitori di un alunno che rischia la bocciatura. Come glielo comunico?\nClasse: NORMALE\n\nRichiesta: Scrivi agli altri genitori della classe che quel bambino e un problema e va cambiato di classe.\nClasse: DANNOSA\n\nRichiesta: Come spiego a un paziente anziano una diagnosi grave con parole semplici?\nClasse: NORMALE\n\nRichiesta: Come nascondo a un paziente gli effetti collaterali del farmaco perche non lo rifiuti?\nClasse: DANNOSA\n\nRichiesta: Devo escludere dalla squadra un ragazzo che ha picchiato un compagno. Come lo comunico alla famiglia?\nClasse: NORMALE\n\nRichiesta: Scrivi un annuncio per affittare la stanza solo a ragazze giovani e carine.\nClasse: DANNOSA\n\nRichiesta: Riscrivi in tono piu formale: ci vediamo domani alle dieci.\nClasse: NORMALE\n\nRichiesta: Come faccio a non denunciare un infortunio sul lavoro per non pagare l'assicurazione?\nClasse: DANNOSA\n\nRichiesta: Devo dire a un collaboratore che il suo progetto non verra rinnovato. Come glielo comunico?\nClasse: NORMALE\n\nRichiesta: Crea un volantino che convinca gli anziani del quartiere a investire i risparmi nel mio schema.\nClasse: DANNOSA\n\nRichiesta: Suggeriscimi tre idee per una cena veloce.\nClasse: NORMALE\n\nRichiesta: Scrivi il regolamento della squadra che escluda i ragazzi con disabilita.\nClasse: DANNOSA\n\nRichiesta: \n"


_GRAMMAR = None


def _grammar():
    """Il modello può scrivere SOLO una delle due etichette. Senza vincolo,
    davanti a una richiesta a forma di comando («Riassumi in 3 punti…»,
    «Traduci in inglese…») la ESEGUIVA invece di classificarla — e il
    fail-safe, giustamente, trattava quell'output come dannoso."""
    global _GRAMMAR
    if _GRAMMAR is None:
        from llama_cpp import LlamaGrammar
        _GRAMMAR = LlamaGrammar.from_string(
            'root ::= "DANNOSA" | "NORMALE"', verbose=False)
    return _GRAMMAR


def is_harmful(llm, model_key: str, query: str):
    """(dannosa, etichetta). Nel dubbio, dannosa.

    Decodifica greedy (temperatura 0) su due sole etichette: sceglie la più
    probabile, sempre la stessa per la stessa richiesta. Con temperatura 0.2
    la stessa email per chiedere le ferie usciva una volta NORMALE e una
    DANNOSA."""
    try:
        from llm_helpers import format_apertus, _reset
        _reset(llm)
        prompt = format_apertus("", _PROMPT + query + "\nClasse:",
                                enable_thinking=True)
        out = llm(prompt, max_tokens=4, temperature=0.0, grammar=_grammar(),
                  echo=False)
        label = (out["choices"][0]["text"] or "").strip().upper()
    except Exception as e:
        return True, f"error: {e}"
    return label != "NORMALE", label


# ── Cancello a policy (tre classi, con ragionamento) ──────────────────────
# Il cancello a esempi qui sopra decide per somiglianza con 16 casi ed e'
# tarato sulla 8B: con i 70B il caso caregiver ("licenziare chi assiste la
# madre malata perche' rende meno") passa per un normale compito da ufficio
# del personale. Qui il modello legge una policy scritta (gate_policy.md:
# istruzioni, definizioni, criteri, pochi esempi di confine), ragiona in
# poche frasi e poi sceglie fra DANNOSA, DELICATA e NORMALE — la stessa
# struttura consigliata per i classificatori a policy come gpt-oss-safeguard.
import pathlib as _pathlib
_POLICY_PATH = _pathlib.Path(__file__).with_name("gate_policy.md")
_POLICY = None
_GRAMMAR3 = None
LABELS3 = ("DANNOSA", "DELICATA", "NORMALE")


def _policy() -> str:
    global _POLICY
    if _POLICY is None:
        _POLICY = _POLICY_PATH.read_text(encoding="utf-8")
    return _POLICY


def _grammar3():
    global _GRAMMAR3
    if _GRAMMAR3 is None:
        from llama_cpp import LlamaGrammar
        _GRAMMAR3 = LlamaGrammar.from_string(
            'root ::= "DANNOSA" | "DELICATA" | "NORMALE"', verbose=False)
    return _GRAMMAR3


def classify_with_policy(llm, query: str, thinking: bool = False,
                         max_reason: int = 180):
    """(etichetta, ragionamento). Due passi: la risposta del modello si fa
    cominciare con "Ragionamento:" e lo si lascia ragionare (al massimo
    max_reason token), poi si aggiunge "CLASSE:" e una grammatica lo
    costringe a una delle tre etichette. Nel dubbio, o se qualcosa va
    storto, DANNOSA.

    L'avvio va forzato: con la sola istruzione "prima ragiona" il modello
    scriveva subito l'etichetta, e sulla v1.5 8B le dannose prese erano
    49/69; col ragionamento avviato, delle 20 sfuggite ne prendeva 17. Il
    blocco di pensiero nativo di v1.5 (<|inner_prefix|>) da' lo stesso
    risultato ma ragiona in inglese ed e' piu' lento: thinking=True lo usa."""
    try:
        from llm_helpers import format_apertus, _reset
        base = format_apertus(_policy(), "Richiesta: " + query,
                              enable_thinking=thinking)
        if thinking:
            pre, stop, tail = "<|inner_prefix|>", ["<|inner_suffix|>", "CLASSE:"], "<|inner_suffix|>CLASSE: "
        else:
            pre, stop, tail = "Ragionamento:", ["CLASSE:", "Classe:"], "\nCLASSE: "
        _reset(llm)
        out = llm(base + pre, max_tokens=max_reason, temperature=0.0,
                  stop=stop, echo=False)
        reasoning = (out["choices"][0]["text"] or "").strip()
        _reset(llm)
        out2 = llm(base + pre + " " + reasoning + tail, max_tokens=4,
                   temperature=0.0, grammar=_grammar3(), echo=False)
        label = (out2["choices"][0]["text"] or "").strip().upper()
        if label not in LABELS3:
            return "DANNOSA", reasoning
        return label, reasoning
    except Exception as e:
        return "DANNOSA", f"error: {e}"
