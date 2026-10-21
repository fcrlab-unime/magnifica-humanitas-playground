#!/usr/bin/env python3
"""
Il documento dietro un brano: per leggere tutto, non solo i 400 caratteri che
la pagina mostra sotto la risposta.

Dato (fonte, riferimento) di un brano recuperato restituisce il testo intero
a cui appartiene, con quel brano segnato, e da dove viene:

  leggi (gdpr, ai_act)  l'articolo (o l'allegato) completo, tutti i suoi
                        brani in ordine, e il link al testo ufficiale di
                        EUR-Lex;
  cei_documenti         il documento completo, il PDF (o doc/rtf) originale
                        se e' stato scaricato all'importazione (raw/), e la
                        pagina del documento su chiesacattolica.it;
  encyclical            l'enciclica, paragrafo per paragrafo.

I testi sono quelli della ricerca (articles.jsonl, chunks.jsonl): quello che
si legge qui e' esattamente quello che il modello aveva davanti.
"""
import json, re
import xml.etree.ElementTree as ET
from project_paths import BASE

CORPUS = BASE / "docs_sample" / "corpus"
ENCYCLICAL = BASE / "docs_sample" / "enciclica" / "chunks.jsonl"
CELEX = {"gdpr": "32016R0679", "ai_act": "32024R1689"}
LABEL = {"gdpr": "GDPR", "ai_act": "AI Act", "cei_documenti": "CEI documents",
         "encyclical": "Magnifica Humanitas",
         "gdpr_recitals": "GDPR — recitals", "ai_act_recitals": "AI Act — recitals"}
_ROWS, _POSTS = {}, None


def _rows(source: str) -> list:
    if source not in _ROWS:
        path = ENCYCLICAL if source == "encyclical" else CORPUS / source / "articles.jsonl"
        _ROWS[source] = [json.loads(l) for l in open(path, encoding="utf-8")]
    return _ROWS[source]


def _posts() -> dict:
    """{id del post: (pagina sul sito, allegato)} dall'export della CEI."""
    global _POSTS
    if _POSTS is None:
        _POSTS = {}
        try:
            root = ET.parse(CORPUS / "cei_documenti" / "source_export.xml").getroot()
            for p in root.findall("post"):
                _POSTS[(p.findtext("id") or "").strip()] = (
                    (p.findtext("Permalink") or "").strip(),
                    (p.findtext("Attachment_URL") or "").strip())
        except Exception:
            pass
    return _POSTS


def attachment(doc_id: str):
    """Il file originale di un documento CEI scaricato all'importazione, o None."""
    m = re.fullmatch(r"CEI-(\d+)", doc_id or "")
    if not m:
        return None
    for path in (CORPUS / "cei_documenti" / "raw").glob(m.group(1) + ".*"):
        if path.suffix.lower() in (".pdf", ".doc", ".docx", ".rtf"):
            return path
    return None


def document(source: str, ref: str):
    """{"title", "label", "parts": [{"ref", "text", "hit"}], "links": [(testo, url)],
    "file": bool} oppure None se fonte o riferimento non esistono."""
    if source not in LABEL:
        return None
    rows = _rows(source)
    links, has_file = [], False
    if source == "encyclical":
        n = re.sub(r"\D", "", ref or "")
        parts = [{"ref": f"§{r['paragraph']}", "text": r["text"], "hit": str(r["paragraph"]) == n}
                 for r in rows]
        title = "Magnifica Humanitas"
    elif source.endswith("_recitals"):
        # Il considerando con i quattro prima e i quattro dopo: da soli si
        # leggono male, tutti e 180 non servono. Il link porta al testo intero.
        m = re.match(r"Considerando (\d+)", ref or "")
        if not m:
            return None
        n = int(m.group(1))
        mine = [r for r in rows if abs(r["recital"] - n) <= 4]
        if not any(r["recital"] == n for r in mine):
            return None
        parts = [{"ref": r["ref"], "text": r["text"].split("\n", 1)[-1], "hit": r["ref"] == ref}
                 for r in mine]
        title = f"{LABEL[source]} {min(r['recital'] for r in mine)}–{max(r['recital'] for r in mine)}"
        links.append(("Official text on EUR-Lex",
                      "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/"
                      f"?uri=CELEX:{CELEX[source.replace('_recitals', '')]}#rct_{n}"))
    elif source == "cei_documenti":
        doc_id = (ref or "").split("§")[0]
        mine = [r for r in rows if r["ref"].split("§")[0] == doc_id]
        if not mine:
            return None
        parts = [{"ref": r["ref"], "text": r["text"], "hit": r["ref"] == ref} for r in mine]
        title = mine[0].get("title") or doc_id
        page, original = _posts().get(doc_id.replace("CEI-", ""), ("", ""))
        has_file = attachment(doc_id) is not None
        if page:
            links.append(("Page on chiesacattolica.it", page))
        if original and not has_file:
            links.append(("Original file on chiesacattolica.it", original))
    else:
        m = re.match(r"Art\. (\d+)", ref or "")
        x = re.match(r"Allegato ([IVXL]+)", ref or "")
        if m:
            mine = [r for r in rows if r.get("article") == int(m.group(1))]
            anchor = f"art_{m.group(1)}"
        elif x:
            mine = [r for r in rows if r.get("annex") == x.group(1)]
            anchor = f"anx_{x.group(1)}"
        else:
            mine = []
        if not mine:
            return None
        # Ogni brano ripete in testa il titolo dell'articolo: qui si legge una volta.
        parts = [{"ref": r["ref"], "text": r["text"].split("\n", 1)[-1], "hit": r["ref"] == ref}
                 for r in mine]
        title = mine[0].get("title") or ref
        links.append(("Official text on EUR-Lex",
                      "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/"
                      f"?uri=CELEX:{CELEX[source]}#{anchor}"))
    if not any(p["hit"] for p in parts):
        parts[0]["hit"] = len(parts) == 1
    return {"title": title, "label": LABEL[source], "parts": parts, "links": links,
            "file": has_file}
