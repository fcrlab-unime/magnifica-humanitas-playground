#!/usr/bin/env python3
"""
Verifica delle citazioni dell'enciclica nelle risposte dei modelli.

Problema: un modello può *inventare* un riferimento ("§103") o citarne uno
sbagliato. Per una tesi serve poter dimostrare che la citazione è reale.

Questo modulo:
  1. carica il testo reale di ogni paragrafo §N dall'enciclica;
  2. estrae i riferimenti §N citati in una risposta;
  3. per ognuno dice se ESISTE nell'enciclica (e ne mostra l'incipit);
  4. opzionalmente verifica se la risposta cita una frase *letterale* del
     paragrafo (groundedness): se il modello mette tra virgolette un testo,
     controlliamo che compaia davvero in un paragrafo dell'enciclica —
  5. e, soprattutto, che compaia nel paragrafo GIUSTO: una citazione reale
     attribuita al § sbagliato ("misattributed") prima passava per valida.

Uso programmatico:
    from cite_verify import verify_response
    report = verify_response(testo_risposta)
    # -> {"refs":[{"ref":"§57","exists":True,"incipit":"..."}],
    #     "quotes":[{"text":"...","status":"verbatim","grounded":True,
    #                "matched_ref":"§103","attributed_ref":"§103"}],
    #     "n_refs":3,"n_invalid":0,"n_quotes":1,
    #     "n_verbatim":1,"n_misattributed":0,"n_paraphrase":0,"n_invented":0}

CLI:
    python cite_verify.py "testo con §57 e «una citazione»"
"""
import json, re, pathlib, sys, os, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import BASE
CHUNKS = BASE / "docs_sample/enciclica/chunks.jsonl"

# Regex: cattura SOLO §57, § 57, §57-58 (range).
# Le forme "punto N" e "par. N" erano accettate ma producevano falsi positivi
# continui: i modelli scrivono elenchi discorsivi ("Ecco una traccia: punto 1.
# ascolta, punto 2. rispondi") e ogni voce veniva contata come citazione
# dell'enciclica — per giunta "valida", perché §1 e §2 esistono davvero.
# Gonfiava proprio la metrica che serve a dire "il framing fa citare di più".
# Verificato su tutte le risposte reali raccolte: i modelli citano SEMPRE e
# solo nella forma §N (le forme a parole non compaiono mai), quindi
# restringere non perde nulla di reale.
_REF_RE = re.compile(r"§\s?(\d{1,3})(?:\s?[-–]\s?(\d{1,3}))?")
# Un singolo riferimento, per capire a quale § una citazione è ATTRIBUITA
# (vedi _attributed_ref): qui i range non servono.
_SINGLE_REF_RE = re.compile(r"§\s?(\d{1,3})")
# Citazioni letterali: testo tra «», "", "" o '' lungo almeno ~25 caratteri.
_QUOTE_RE = re.compile(r"[«\"“‘]([^»\"”’]{25,400})[»\"”’]")
# Il virgolettato è presentato come CITAZIONE DELLA FONTE? Serve perché il
# modello mette fra virgolette soprattutto testo suo — la bozza di email che
# propone, l'apertura consigliata — e quello non è una citazione mancata:
# non pretende di esserlo. Nelle condizioni "neutral" però le citazioni vere
# arrivano senza numero (il manifesto neutro vieta i §), quindi non basta
# cercare il §: vale anche il rimando esplicito alla fonte poco prima.
_CLAIM_RE = re.compile(r"\b(enciclica|magnifica\s+humanitas|manifesto)\b",
                       re.IGNORECASE)


def _norm(s: str) -> str:
    """Normalizza per il match testuale: minuscole, niente accenti/punteggiatura
    multipla, spazi compattati. Rende il confronto robusto a piccole differenze."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Stopword italiane comuni: escluse dal confronto "parole-chiave" per la
# rilevazione delle parafrasi (altrimenti articoli/preposizioni gonfiano il match).
_STOP = {
    "il","lo","la","i","gli","le","un","uno","una","del","dello","della","dei",
    "degli","delle","di","a","da","in","con","su","per","tra","fra","al","allo",
    "alla","ai","agli","alle","dal","dalla","nel","nella","nei","sul","sulla",
    "non","che","chi","cui","e","ed","o","od","ma","se","come","anche","più",
    "puo","puoi","può","essere","viene","sono","è","ha","ho","hai","si","ci",
    "questo","questa","quello","quella","suo","sua","loro","ogni","tutto","tutti",
}

def _content_stems(norm_text: str) -> set:
    """Insieme delle parole-contenuto (len≥4, non stopword) ridotte allo stem
    (primi 5 caratteri) → robusto alle flessioni (considerare/considerata)."""
    return {t[:5] for t in norm_text.split() if len(t) >= 4 and t not in _STOP}


def _verbatim_in(qn: str, pnum: str) -> bool:
    """La citazione normalizzata `qn` compare (anche con piccole omissioni)
    nel paragrafo `pnum`? Stessa soglia usata per la ricerca globale."""
    ptxt = _PARAS_NORM.get(pnum)
    if not ptxt:
        return False
    if qn in ptxt:
        return True
    toks = qn.split()
    if len(toks) < 4:
        return False
    grams = {" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)}
    return sum(1 for g in grams if g in ptxt) / len(grams) >= 0.6


def _clause_before(text: str, start: int, span: int = 160) -> str:
    """Il pezzo di frase che INTRODUCE il virgolettato. Si ferma alla fine
    della frase precedente o alla chiusura di un virgolettato precedente:
    senza questo taglio, un § (o la parola "Enciclica") appartenente alla
    frase prima verrebbe scambiato per l'introduzione di QUESTA citazione —
    con l'effetto di attribuire a una bozza il § di una citazione vera, e
    quindi di accusare di misattribuzione del testo che non cita nulla."""
    w = text[max(0, start - span):start]
    cut = 0
    for ch in (".", "!", "?", "\n", "»", "”", "\"", "’"):
        cut = max(cut, w.rfind(ch) + 1)
    return w[cut:]


def _clause_after(text: str, end: int, span: int = 60) -> str:
    """Il pezzo di frase che SEGUE il virgolettato, fino a fine frase."""
    w = text[end:end + span]
    m = re.search(r"[.!?\n]", w)
    return w[:m.start()] if m else w


def _attributed_ref(text: str, start: int, end: int):
    """Il § a cui la citazione è ATTRIBUITA dal testo: il riferimento nella
    frase che la introduce (es. «Come ricorda l'Enciclica (§150), "…"»),
    altrimenti quello che la segue nella stessa frase (es. «"…" (§214)»).
    None se la citazione non è accompagnata da nessun riferimento."""
    ms = list(_SINGLE_REF_RE.finditer(_clause_before(text, start)))
    if ms:
        return ms[-1].group(1)
    m = _SINGLE_REF_RE.search(_clause_after(text, end))
    return m.group(1) if m else None


def _is_citation_claim(text: str, start: int, attributed) -> bool:
    """Il virgolettato pretende di essere una citazione della fonte?
    Sì se porta un § accanto, o se la frase che lo introduce nomina
    l'enciclica («Come ricorda l'Enciclica, "…"» — forma tipica delle
    condizioni neutre, dove i numeri di paragrafo sono vietati). Altrimenti
    è testo che il modello propone lui (la bozza di risposta, una frase
    d'apertura consigliata) e non va giudicato come citazione."""
    if attributed:
        return True
    return bool(_CLAIM_RE.search(_clause_before(text, start)))


_PARAS = None        # {"57": "testo…"}
_PARAS_NORM = None    # {"57": "testo normalizzato"}


def _load():
    global _PARAS, _PARAS_NORM
    if _PARAS is not None:
        return
    _PARAS, _PARAS_NORM = {}, {}
    if not CHUNKS.exists():
        return
    for line in open(CHUNKS, encoding="utf-8"):
        c = json.loads(line)
        num = str(c["id"]).lstrip("§").strip()
        txt = c["text"]
        # Più chunk possono condividere lo stesso § (raro): concateniamo.
        _PARAS[num] = (_PARAS.get(num, "") + " " + txt).strip()
    for num, txt in _PARAS.items():
        _PARAS_NORM[num] = _norm(txt)


def _extract_refs(text: str):
    """Lista ordinata e deduplicata dei numeri di paragrafo citati (espande i range)."""
    nums = []
    for m in _REF_RE.finditer(text):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b - a > 30:          # range improbabile → tratta come singolo
            b = a
        for n in range(a, b + 1):
            if n not in nums:
                nums.append(n)
    return nums


def verify_response(text: str, check_quotes: bool = True) -> dict:
    """Verifica i riferimenti §N e (opzionale) le citazioni letterali."""
    _load()
    refs_out = []
    for n in _extract_refs(text):
        key = str(n)
        exists = key in _PARAS
        refs_out.append({
            "ref": f"§{n}",
            "exists": exists,
            "incipit": (_PARAS[key][:90] + "…") if exists else None,
        })

    quotes_out = []
    if check_quotes:
        for m in _QUOTE_RE.finditer(text):
            q = m.group(1).strip()
            qn = _norm(q)
            if len(qn) < 20:
                continue
            # Quattro livelli: verbatim (citazione esatta, attribuita al §
            # giusto) · misattributed (testo reale, ma attribuito al §
            # sbagliato) · paraphrase (concetto reale ma riscritto, spacciato
            # per citazione) · invented (nessun riscontro).
            # La misattribuzione è il caso che prima sfuggiva del tutto: si
            # controllava che il § esistesse E che la citazione esistesse da
            # qualche parte, mai che le due cose combaciassero. Su una
            # risposta reale il modello citava "(§150)" e "(§103)" mettendo
            # però tra virgolette il testo di §105, e il verdetto era
            # "all verified" — esattamente il falso via libera che una tesi
            # non si può permettere.
            attributed = _attributed_ref(text, m.start(), m.end())
            claimed = _is_citation_claim(text, m.start(), attributed)
            status, matched = "invented", None

            # 1) VERBATIM attribuito bene — la citazione è DAVVERO nel §
            #    indicato dal testo. Controllato per primo: una stessa frase
            #    può comparire in più paragrafi, e se quello dichiarato è fra
            #    questi l'attribuzione è corretta, non "sbagliata per caso".
            if attributed and _verbatim_in(qn, attributed):
                status, matched = "verbatim", attributed
            # 2) VERBATIM altrove — testo reale, ma non nel § dichiarato
            if not matched:
                best, best_frac = None, 0.0
                toks = qn.split()
                grams = ({" ".join(toks[i:i+4]) for i in range(len(toks)-3)}
                         if len(toks) >= 4 else set())
                for num, ptxt in _PARAS_NORM.items():
                    if qn in ptxt:
                        best, best_frac = num, 1.0
                        break
                    if grams:
                        frac = sum(1 for g in grams if g in ptxt) / len(grams)
                        if frac > best_frac:
                            best, best_frac = num, frac
                if best_frac >= 0.6:
                    matched = best
                    status = "misattributed" if attributed else "verbatim"
            # 3) PARAPHRASE — non letterale, ma le parole-chiave della citazione
            #    compaiono quasi tutte in uno stesso paragrafo (concetto reale).
            #    Solo per chi pretende di citare: su testo proprio del modello
            #    un match di parole-chiave non significa niente.
            if not matched and claimed:
                best, best_rec = None, 0.0
                qstems = _content_stems(qn)
                if len(qstems) >= 2:
                    for num, ptxt in _PARAS_NORM.items():
                        pstems = _content_stems(ptxt)
                        if not pstems:
                            continue
                        rec = len(qstems & pstems) / len(qstems)
                        if rec > best_rec:
                            best, best_rec = num, rec
                    if best_rec >= 0.7:
                        status, matched = "paraphrase", best
            # 4) Nessun riscontro e nessuna pretesa di citare: è testo del
            #    modello (una bozza, una frase consigliata). Non è una
            #    citazione inventata — non si spacciava per una citazione.
            if not matched and not claimed:
                status = "not_claimed"
            quotes_out.append({
                "text": q[:120] + ("…" if len(q) > 120 else ""),
                "status": status,
                "grounded": status == "verbatim",   # retro-compat
                "claimed": claimed,
                "matched_ref": f"§{matched}" if matched else None,
                # § dichiarato dal testo accanto alla citazione (None se la
                # citazione non è accompagnata da nessun riferimento).
                "attributed_ref": f"§{attributed}" if attributed else None,
            })

    return {
        "refs": refs_out,
        "quotes": quotes_out,
        "n_refs": len(refs_out),
        "n_invalid": sum(1 for r in refs_out if not r["exists"]),
        "n_quotes": len(quotes_out),
        "n_verbatim":      sum(1 for q in quotes_out if q["status"] == "verbatim"),
        "n_misattributed": sum(1 for q in quotes_out if q["status"] == "misattributed"),
        "n_paraphrase":    sum(1 for q in quotes_out if q["status"] == "paraphrase"),
        "n_invented":      sum(1 for q in quotes_out if q["status"] == "invented"),
        # virgolettati che NON si presentano come citazioni (bozze, frasi
        # proposte dal modello): contati a parte, non sono errori
        "n_not_claimed":   sum(1 for q in quotes_out if q["status"] == "not_claimed"),
        # denominatore onesto per le percentuali: solo chi pretende di citare
        "n_claimed":       sum(1 for q in quotes_out if q["claimed"]),
        # non-verbatim (misattribuite + parafrasi + inventate): nome storico
        "n_ungrounded": sum(1 for q in quotes_out if q["status"] != "verbatim"),
    }


def _ref_spans(text: str, start: int, end: int):
    """(inizio, fine, numero) del § a cui la citazione in text[start:end] e'
    attribuita, nel testo intero: stessa regola di _attributed_ref."""
    clause = _clause_before(text, start)
    off = start - len(clause)
    ms = list(_SINGLE_REF_RE.finditer(clause))
    if ms:
        m = ms[-1]
        return off + m.start(), off + m.end(), m.group(1)
    after = _clause_after(text, end)
    m = _SINGLE_REF_RE.search(after)
    if m:
        return end + m.start(), end + m.end(), m.group(1)
    return None


def _ref_token_span(text: str, span) -> tuple:
    """(inizio, fine, "") per togliere un § con le parentesi che lo
    racchiudono e lo spazio prima: "(§74)" o " §74", non solo "§74"."""
    s, e = span[0], span[1]
    if s > 0 and text[s - 1] == "(" and e < len(text) and text[e] == ")":
        s, e = s - 1, e + 1
    while s > 0 and text[s - 1] == " ":
        s -= 1
    return s, e, ""


def _sentence_span(text: str, start: int, end: int):
    """La frase che contiene text[start:end]."""
    s = max(text.rfind(ch, 0, start) for ch in (".", "!", "?", "\n")) + 1
    m = re.search(r"[.!?\n]", text[end:])
    e = end + (m.end() if m else len(text) - end)
    return s, e


def repair_citations(text: str):
    """Rende vincolante la verifica: invece di segnalare le citazioni
    sbagliate, le corregge prima che l'utente le veda. (testo, riepilogo).

    - attribuita al § sbagliato (il testo e' vero): si corregge il numero;
    - parafrasi fra virgolette (concetto reale, parole no): via le
      virgolette, § corretto;
    - inventata (nessun riscontro nel testo): via la frase intera, perche'
      attribuirebbe all'enciclica cio' che non dice;
    - § che non esiste: via il riferimento.
    Misurato sulle risposte di Apertus 70B, che scrive con parole sue e
    mette fra virgolette anche le parafrasi: 73 citazioni sbagliate su 108.
    """
    _load()
    fixed = {"misattributed": 0, "paraphrase": 0, "invented": 0, "invalid_ref": 0}
    edits = []   # (inizio, fine, sostituzione), applicati da destra
    for m in _QUOTE_RE.finditer(text):
        q = m.group(1).strip()
        qn = _norm(q)
        if len(qn) < 20:
            continue
        single = verify_response(_surround(text, m), check_quotes=True)
        info = next((x for x in single["quotes"] if _norm(x["text"].rstrip("…"))[:60] == qn[:60]), None)
        if not info:
            continue
        st, matched = info["status"], (info["matched_ref"] or "").lstrip("§")
        span = _ref_spans(text, m.start(), m.end())
        if st == "misattributed" and span and matched:
            edits.append((span[0], span[1], f"§{matched}"))
            fixed["misattributed"] += 1
        elif st == "paraphrase":
            edits.append((m.start(), m.end(), q))            # via le virgolette
            # Il numero si toglie, non si sostituisce: la parafrasi e'
            # riconosciuta per parole in comune, e su frasi generiche
            # ("ogni persona ha diritto alla dignita' e al lavoro") il
            # paragrafo "piu' simile" e' un tiro a sorte.
            if span and span[2] != matched:
                edits.append(_ref_token_span(text, span))
            fixed["paraphrase"] += 1
        elif st == "invented":
            s, e = _sentence_span(text, m.start(), m.end())
            edits.append((s, e, ""))
            fixed["invented"] += 1
    # § inesistenti fuori dalle citazioni gia' sistemate
    for m in _SINGLE_REF_RE.finditer(text):
        if m.group(1) not in _PARAS:
            edits.append(_ref_token_span(text, (m.start(), m.end(), m.group(1))))
            fixed["invalid_ref"] += 1
    # applica da destra, saltando le modifiche che si sovrappongono a una
    # gia' applicata (es. un § dentro una frase inventata gia' tolta)
    out, last = text, len(text) + 1
    for s, e, rep in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        if e > last:
            continue
        out = out[:s] + rep + out[e:]
        last = s
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +([.,;:])", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, fixed


def _surround(text: str, m) -> str:
    """La citazione con la frase che la introduce e quella che la segue:
    abbastanza contesto perche' verify_response la giudichi come nel testo
    intero, ma da sola (senza le altre citazioni della risposta)."""
    s = m.start() - len(_clause_before(text, m.start()))
    e = m.end() + len(_clause_after(text, m.end()))
    return text[s:e]


if __name__ == "__main__":
    txt = sys.argv[1] if len(sys.argv) > 1 else (
        'Non posso. Come ricorda il §57 sui vulnerabili, e il §103 avverte '
        'contro «affidare a un algoritmo il potere di selezionare chi merita». '
        'Vedi anche il §999 che non esiste e «una citazione totalmente inventata».'
    )
    rep = verify_response(txt)
    print(f"\nRiferimenti trovati: {rep['n_refs']} (invalidi: {rep['n_invalid']})")
    for r in rep["refs"]:
        mark = "✓" if r["exists"] else "✗ INVENTATO"
        print(f"  {mark} {r['ref']}" + (f"  — {r['incipit']}" if r["incipit"] else ""))
    print(f"\nVirgolettati: {rep['n_quotes']} (di cui {rep['n_claimed']} presentati "
          f"come citazioni)\n  verbatim {rep['n_verbatim']} · "
          f"misattribuite {rep['n_misattributed']} · parafrasi {rep['n_paraphrase']} · "
          f"inventate {rep['n_invented']} · non-citazioni {rep['n_not_claimed']}")
    MARK = {"verbatim": "✓ VERBATIM", "misattributed": "✗ ATTRIBUITA MALE",
            "paraphrase": "◐ PARAFRASI di", "invented": "✗ INVENTATA",
            "not_claimed": "· testo del modello, non una citazione:"}
    for q in rep["quotes"]:
        if q["status"] == "misattributed":
            detail = f" — dichiarata {q['attributed_ref']}, in realtà {q['matched_ref']}"
        else:
            detail = f" {q['matched_ref']}" if q["matched_ref"] else ""
        print(f"  {MARK[q['status']]}{detail}  «{q['text']}»")
