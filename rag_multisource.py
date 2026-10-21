#!/usr/bin/env python3
"""
RAG multi-corpus: indicizza l'enciclica + tutti i corpus normativi trovati
in docs_sample/corpus/*/articles.jsonl, in ChromaDB con metadati `source`
per filtrare le query. Le fonti sono scoperte dinamicamente (basta
aggiungere o togliere una cartella in docs_sample/corpus/, senza toccare
il codice) — vedi discover_corpus_sources().

Riusa il vector store esistente (`docs_sample/enciclica/chroma_db`) ma aggiunge
una collection nuova `multisource` che contiene TUTTI i chunks.
La collection esistente `magnifica_humanitas` rimane intatta (compat retro).

Usage:
    python rag_multisource.py index             # costruisce/ricostruisce
    python rag_multisource.py query "..."        # test (top-3 da tutto)
    python rag_multisource.py query --source gdpr "..."  # filtra per source
"""
import json, pathlib, argparse, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import BASE
CORPUS_DIR = BASE / "docs_sample" / "corpus"
ENCY_CHUNKS = BASE / "docs_sample" / "enciclica" / "chunks.jsonl"
CHROMA_DIR = BASE / "docs_sample" / "enciclica" / "chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-large"
COLLECTION = "multisource"


# Etichette/colori "curati" per le fonti note; una cartella in docs_sample/corpus/
# senza entry qui viene comunque indicizzata, con un'etichetta derivata dal nome
# della cartella (vedi list_sources()).
SOURCE_META = {
    "encyclical":    {"label":"Magnifica Humanitas", "icon":"🕊️", "color":"#1B5E20"},
    "gdpr":          {"label":"GDPR (Reg. UE 2016/679)", "icon":"🔒", "color":"#0288D1"},
    "ai_act":        {"label":"AI Act (Reg. UE 2024/1689)", "icon":"🤖", "color":"#7B1FA2"},
    "cei_documenti": {"label":"CEI Secretariat documents", "icon":"⛪", "color":"#8B4513"},
    "statuto":    {"label":"Statuto dei Lavoratori (L. 300/1970)", "icon":"⚖️", "color":"#EF6C00"},
}


def discover_corpus_sources() -> list[str]:
    """Cartelle in docs_sample/corpus/ che contengono un articles.jsonl."""
    if not CORPUS_DIR.exists():
        return []
    return sorted(p.parent.name for p in CORPUS_DIR.glob("*/articles.jsonl"))


def list_sources() -> list[dict]:
    """Fonti RAG disponibili (enciclica + ogni corpus scoperto), con conteggio
    chunk ed etichetta/colore per la UI. Usata da web_app.py per generare i
    checkbox delle sorgenti senza hardcodare nulla."""
    out = []
    if ENCY_CHUNKS.exists():
        n = sum(1 for _ in open(ENCY_CHUNKS, encoding="utf-8"))
        meta = SOURCE_META.get("encyclical", {})
        out.append({"key": "encyclical",
                    "label": meta.get("label", "Magnifica Humanitas"),
                    "color": meta.get("color", "#1B5E20"), "count": n})
    for src in discover_corpus_sources():
        n = sum(1 for _ in open(CORPUS_DIR / src / "articles.jsonl", encoding="utf-8"))
        meta = SOURCE_META.get(src, {})
        label = meta.get("label") or src.replace("_", " ").replace("-", " ").title()
        out.append({"key": src, "label": label,
                    "color": meta.get("color", "#374151"), "count": n})
    return out


def load_all_chunks():
    """Carica tutti i chunks dei 4 corpus in un'unica lista normalizzata."""
    out = []
    # 1. Enciclica (formato esistente)
    if ENCY_CHUNKS.exists():
        for line in open(ENCY_CHUNKS, encoding="utf-8"):
            c = json.loads(line)
            out.append({
                "source": "encyclical",
                "id":     f"encyclical:{c['id']}",
                "ref":    c["id"],            # già '§N'
                "title":  f"{c['chapter']} — {c['section']}",
                "chapter": c.get("chapter",""),
                "text":   c["text"],
            })
        print(f"  • encyclical: {sum(1 for _ in open(ENCY_CHUNKS))} chunks")
    # 2-N. Corpus normativi (una cartella per fonte, scoperte dinamicamente)
    for src in discover_corpus_sources():
        path = CORPUS_DIR / src / "articles.jsonl"
        n = 0
        for line in open(path, encoding="utf-8"):
            c = json.loads(line)
            out.append({
                "source": src,
                # Un comma diviso per lettere da' piu' brani con lo stesso
                # riferimento (import_eurlex): "part" li distingue.
                "id":     f"{src}:{c['ref']}" + (f"#{c['part']}" if "part" in c else ""),
                "ref":    c["ref"],
                "title":  c.get("title", c["ref"]),
                "chapter": "",
                "text":   c["text"],
            })
            n += 1
        print(f"  • {src}: {n} chunks")
    return out


def build_index():
    print("◦ Loading chunks from all corpora…")
    chunks = load_all_chunks()
    if not chunks:
        print("  ✗ No chunks!"); return
    print(f"  Total: {len(chunks)} chunks\n")

    print(f"◦ Loading embeddings model: {EMBED_MODEL}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    print("◦ Computing embeddings (batch=16)…")
    texts = [f"passage: {c['text'][:1800]}" for c in chunks]  # cap per stabilità
    emb = model.encode(texts, batch_size=16, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    print(f"  ✓ shape {emb.shape}")

    print(f"◦ Saving to ChromaDB: {CHROMA_DIR}/{COLLECTION}")
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(name=COLLECTION,
                                    metadata={"hnsw:space": "cosine"})
    # ChromaDB limita la dimensione di ogni .add() (es. ~5461 col backend
    # sqlite/rust attuale); con corpus grandi (>5000 chunk, es. cei_documenti)
    # serve inserire a lotti.
    ids = [c["id"] for c in chunks]
    docs = [c["text"][:6000] for c in chunks]
    embs = emb.tolist()
    metas = [{"source": c["source"], "ref": c["ref"],
             "title": c["title"], "chapter": c["chapter"]} for c in chunks]
    try:
        batch_size = client.get_max_batch_size()
    except Exception:
        batch_size = 2000
    for i in range(0, len(chunks), batch_size):
        j = min(i + batch_size, len(chunks))
        coll.add(ids=ids[i:j], documents=docs[i:j],
                 embeddings=embs[i:j], metadatas=metas[i:j])
        print(f"    … inserted {j}/{len(chunks)}")
    print(f"  ✓ {coll.count()} chunks indexed")


# Cross-encoder reranker: riordina i candidati per pertinenza reale alla query.
# L'embedding bi-encoder è veloce ma approssimato; il cross-encoder legge
# (query, passaggio) insieme e dà uno score di rilevanza molto più preciso.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# ─── Retriever multi-source ──────────────────────────────────────────────────
class MultiSourceRetriever:
    def __init__(self, use_reranker: bool = True):
        from sentence_transformers import SentenceTransformer
        import chromadb
        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.coll = client.get_collection(COLLECTION)
        self.reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            except Exception as e:
                print(f"  ⚠ Reranker not loaded ({e}); using the bi-encoder only")

    def _query_pool(self, qemb, query: str, where, k: int,
                    min_similarity: float, overfetch: int) -> list:
        """Interroga il vector store (con un filtro `where` opzionale) e
        restituisce i top-k, rerankati se disponibile un cross-encoder."""
        n_fetch = max(k, overfetch) if self.reranker else k * 2
        result = self.coll.query(query_embeddings=qemb.tolist(),
                                  n_results=n_fetch, where=where)
        cand = []
        for i, doc_id in enumerate(result["ids"][0]):
            similarity = 1 - result["distances"][0][i]
            md = result["metadatas"][0][i]
            cand.append({
                "id":         doc_id,
                "source":     md.get("source"),
                "ref":        md.get("ref"),
                "title":      md.get("title"),
                "chapter":    md.get("chapter") or "",
                "text":       result["documents"][0][i],
                "similarity": round(similarity, 3),
            })

        # ── Reranking col cross-encoder ──────────────────────────────────────
        if self.reranker and cand:
            pairs = [(query, c["text"]) for c in cand]
            scores = self.reranker.predict(pairs)
            for c, s in zip(cand, scores):
                c["rerank_score"] = float(s)
            cand.sort(key=lambda c: c["rerank_score"], reverse=True)
            # Il cross-encoder bge dà logit; sigmoide >0 ≈ rilevante.
            out = [c for c in cand if c["rerank_score"] > -2.0][:k]
            if not out:  # fallback: tieni comunque i top-k
                out = cand[:k]
            return out

        # ── Senza reranker: filtro per similarità bi-encoder ─────────────────
        return [c for c in cand if c["similarity"] >= min_similarity][:k]

    def retrieve(self, query: str, k: int = 3,
                 sources: list = None,
                 min_similarity: float = 0.4,
                 overfetch: int = 10,
                 source_quota: int = 1,
                 encyclical_query: str = None) -> list:
        """
        source_quota: quando sono selezionate PIÙ fonti insieme, ciascuna si
        riserva sempre almeno source_quota posti (interrogata da sola, non
        mescolata nel pool con le altre) prima che il resto del budget k
        vada a un ranking libero su tutte le fonti selezionate. Senza
        questo, una fonte piccola (encyclical 236 chunk, gdpr 99, ai_act
        113) viene sistematicamente sommersa da corpus molto più grandi
        come cei_documenti (12596 chunk, il 96.6% del corpus combinato
        quando tutte le fonti sono selezionate): a parità di query, un
        corpus enorme ha statisticamente più probabilità di produrre
        QUALCHE chunk con similarità/rerank score alto anche se meno
        pertinente in senso stretto, semplicemente per il numero di
        candidati fra cui pescare — non è che la fonte piccola sia "meno
        rilevante", è sommersa dal volume. Con la quota, ogni fonte scelta
        resta sempre rappresentata; il resto del budget va a ranking
        libero su tutte, quindi una fonte può comunque prendersi più posti
        se è davvero la più pertinente.

        encyclical_query: se diverso da None, usato SOLO per la quota
        garantita dell'enciclica (embedding + reranking), al posto di
        `query`. Serve a chi chiama per applicare la query expansion
        (pensata per il lessico dell'enciclica: concetti-ponte etici) solo
        lì, senza inquinare la ricerca sulle altre fonti — verificato che
        una domanda GDPR come "posso conservare i dati oltre 5 anni?" con
        l'espansione finisce per cercare anche "delega decisionale
        all'algoritmo" (falso positivo su "selezione candidati" → pattern
        pensato per lo scoring algoritmico), rumore per una ricerca legale.
        """
        sel = sources or []
        use_quota = len(sel) > 1 and source_quota > 0

        if not use_quota:
            qemb = self.model.encode([f"query: {query}"],
                                     normalize_embeddings=True,
                                     convert_to_numpy=True)
            where = None
            if sources:
                where = ({"source": sources[0]} if len(sources) == 1
                         else {"$or": [{"source": s} for s in sources]})
            return self._query_pool(qemb, query, where, k, min_similarity, overfetch)

        # ── Retrieval stratificato: ogni fonte selezionata garantita, il
        #    resto del budget va a ranking libero su tutte ────────────────────
        qemb = self.model.encode([f"query: {query}"],
                                 normalize_embeddings=True,
                                 convert_to_numpy=True)
        guaranteed, seen_ids = [], set()
        for src in sel:
            if len(guaranteed) >= k:
                break
            if src == "encyclical" and encyclical_query:
                src_query = encyclical_query
                src_qemb = self.model.encode([f"query: {src_query}"],
                                             normalize_embeddings=True,
                                             convert_to_numpy=True)
            else:
                src_query, src_qemb = query, qemb
            got = self._query_pool(src_qemb, src_query, {"source": src},
                                   min(source_quota, k - len(guaranteed)),
                                   min_similarity, overfetch)
            guaranteed.extend(got)
            seen_ids.update(c["id"] for c in got)

        remaining_k = k - len(guaranteed)
        if remaining_k <= 0:
            return guaranteed
        where_all = {"$or": [{"source": s} for s in sel]}
        # Overfetch maggiorato perché i già-garantiti vanno scartati dal pool.
        rest = self._query_pool(qemb, query, where_all,
                                remaining_k + len(seen_ids), min_similarity,
                                overfetch + len(seen_ids))
        rest = [c for c in rest if c["id"] not in seen_ids][:remaining_k]
        return guaranteed + rest


# ─── Composer dei prompt RAG multi-source ───────────────────────────────────
# Istruzione comune che impone rifiuto motivato per le richieste dannose.
_RAG_PROTOCOL = (
    "Come usare questi riferimenti:\n"
    "1. Valuta prima se la richiesta dell'utente, se eseguita alla lettera, "
    "lederebbe la dignità di una persona, violerebbe un diritto, sarebbe "
    "discriminatoria o userebbe un linguaggio che umilia qualcuno.\n"
    "2. SE SÌ: non produrre ciò che è stato chiesto. Dichiara con chiarezza "
    "che non puoi farlo e SPIEGA IL PERCHÉ in modo discorsivo e riferito a "
    "questo caso concreto, appoggiandoti ai passaggi qui sopra (spiega cosa "
    "dicono e come si applicano). Poi proponi alternative concrete e "
    "praticabili per raggiungere un obiettivo legittimo senza il danno.\n"
    "3. SE la richiesta è solo delicata ma legittima: NON aprire con «Non "
    "posso» (riservato SOLO al caso dannoso sopra) — assisti direttamente, "
    "fornendo quello che è stato chiesto, premettendo i rischi e come "
    "mitigarli.\n"
    "4. Spiega sempre il *perché* delle tue raccomandazioni: l'utente deve "
    "capire il ragionamento, non ricevere solo un sì o un no.\n\n"
)

def compose_neutral(chunks: list, user_query: str) -> str:
    if not chunks:
        return user_query
    passages = "\n\n".join(
        f"[Riferimento {i+1}] {c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        "Considera questi riferimenti normativi ed etici come linee guida "
        "vincolanti per la tua risposta:\n\n"
        f"{passages}\n\n"
        f"{_RAG_PROTOCOL}"
        f"Richiesta dell'utente: {user_query}"
    )

def compose_explicit(chunks: list, user_query: str) -> str:
    if not chunks:
        return user_query
    passages = []
    for c in chunks:
        src_meta = SOURCE_META.get(c["source"], {"label": c["source"]})
        passages.append(
            f"[{src_meta['label']} — {c['ref']}"
            + (f", {c['chapter']}" if c["chapter"] else "")
            + (f", {c['title']}" if c.get("title") and c["title"]!=c["ref"] else "")
            + "]\n" + c["text"]
        )
    body = "\n\n".join(passages)
    return (
        "Considera questi passaggi tratti dalle fonti normative ed etiche "
        "europee e italiane. Sono vincolanti per la tua risposta e devi "
        "citarli esplicitamente quando spieghi il tuo ragionamento:\n\n"
        f"{body}\n\n"
        f"{_RAG_PROTOCOL}"
        f"Richiesta dell'utente: {user_query}"
    )


# ─── CLI ────────────────────────────────────────────────────────────────────
def cmd_query(text, sources=None, k=3):
    r = MultiSourceRetriever()
    chunks = r.retrieve(text, k=k, sources=sources)
    if not chunks:
        print(f"  No relevant passage (sim < 0.4)"); return
    print(f"\n  Top-{len(chunks)} results"
          + (f" (filtro: {sources})" if sources else "")
          + ":")
    for c in chunks:
        src_l = SOURCE_META.get(c["source"], {}).get("label", c["source"])
        print(f"\n  ── [{src_l}] {c['ref']}  · sim {c['similarity']}")
        if c["title"]:
            print(f"     {c['title']}")
        print(f"     {c['text'][:250]}…")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["index","query"])
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("--source", action="append", default=None,
                   help="Filtra per source key (vedi list_sources()), ripetibile")
    p.add_argument("-k", type=int, default=3)
    args = p.parse_args()
    if args.cmd == "index":
        build_index()
    elif args.cmd == "query":
        if not args.text:
            print("Specify a query"); sys.exit(1)
        cmd_query(args.text, sources=args.source, k=args.k)
