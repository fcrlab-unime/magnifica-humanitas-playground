#!/usr/bin/env python3
"""
Query expansion per il RAG sull'enciclica.

Problema: le richieste degli utenti sono CONCRETE ("licenziare un caregiver",
"monitorare i dipendenti"), mentre l'enciclica usa un linguaggio ASTRATTO
("dignità del lavoro", "custodia della libertà", "sorveglianza pervasiva").
Questo mismatch lessicale fa sì che il retrieval semantico a volte manchi i
passaggi più pertinenti.

Soluzione: prima di interrogare il vector store, arricchiamo la query con i
"concetti-ponte" astratti dell'enciclica attivati da parole concrete presenti
nella richiesta. La query arricchita serve SOLO al retrieval; al modello resta
mostrata la richiesta originale.

Approccio deterministico (mappa concetto→termini), non un LLM: è veloce,
riproducibile e trasparente — qualità desiderabili per una tesi.
"""
import re

# Ogni voce: (regex sui termini concreti, lista di concetti astratti dell'enciclica)
_CONCEPT_MAP = [
    (r"licenzi|esuber|rendiment|prestazion|lavorator|dipendent|impieg|assun|contratt",
     ["dignità del lavoro", "il lavoratore non va ridotto a una metrica",
      "disoccupazione", "tutela dell'occupazione"]),
    (r"sorveglian|monitor|controll|tracc|telecamer|geolocalizz|spia",
     ["sorveglianza pervasiva", "custodia della libertà",
      "controllo sociale", "nuove schiavitù", "dipendenza"]),
    (r"discrimin|gener[ei]|sess|etni|razz|origin|religion|età|disabil|orientament",
     ["stereotipi e discriminazione", "giustizia sociale",
      "dignità inalienabile della persona", "diritti delle minoranze"]),
    (r"dat[oi] personal|privacy|profila|gdpr|sensibil|consens",
     ["diritto alla privacy", "diritti inviolabili della persona",
      "protezione dei dati"]),
    (r"algoritm|automat|decision|selezion|scart|punteggi|scoring|classific",
     ["delega decisionale all'algoritmo", "responsabilità e accountability",
      "il giudizio umano non sostituibile", "apparente oggettività dell'IA"]),
    (r"minor|bambin|anzian|malat|migrant|fragil|povert|vulnerabil|disabil",
     ["protezione dei più vulnerabili", "standard di cura più alto",
      "diritti dei fragili"]),
    (r"umili|aggressiv|insult|denigr|deride|offens|minacc|ricatt",
     ["ecologia della comunicazione", "disarmare le parole",
      "linguaggio che rispetta la dignità"]),
    (r"profitt|cost[oi]|efficienz|taglio|risparmi|margin|fatturat|produttiv",
     ["bene comune", "cura prima della potenza",
      "giustizia sociale", "non sacrificare le persone al profitto"]),
    (r"manipol|persuad|inganno|disinform|fake|propagand|polariz",
     ["cura della verità", "verità come bene comune",
      "trasparenza e non manipolazione"]),
    (r"relazion|solitudin|compagn|amic|chatbot|sostitu|affett",
     ["cura dei legami umani", "non sostituire le relazioni umane con l'IA"]),
]

# Ancore sempre aggiunte: il cuore dell'enciclica.
_ANCHORS = ["dignità della persona umana", "centralità della persona"]


def expand_query(query: str, max_terms: int = 8) -> tuple[str, list]:
    """Restituisce (query_arricchita, lista_concetti_aggiunti).

    La query arricchita è 'query originale + concetti', usata solo per il
    retrieval. Se nessun trigger scatta, aggiunge solo le ancore.
    """
    q = query.lower()
    added = []
    for pattern, concepts in _CONCEPT_MAP:
        if re.search(pattern, q):
            for c in concepts:
                if c not in added:
                    added.append(c)
    # Aggiungi le ancore se c'è spazio (utili anche quando nessun trigger scatta).
    for a in _ANCHORS:
        if a not in added:
            added.append(a)
    added = added[:max_terms]
    expanded = query + " — temi correlati: " + "; ".join(added)
    return expanded, added


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "Devo licenziare Mario, rende poco, anche se assiste la madre malata"
    exp, terms = expand_query(q)
    print("ORIGINALE:", q)
    print("\nCONCETTI AGGIUNTI:")
    for t in terms:
        print("  •", t)
    print("\nQUERY ARRICCHITA (per il retrieval):\n ", exp)
