#!/usr/bin/env python3
"""
Banco di esempi few-shot, scelti per somiglianza con la richiesta reale.

Perché esiste
-------------
Apertus 8B non applica regole astratte: misurato su una batteria di 12 casi,
ha ignorato i 18 principi del manifesto, una griglia operativa con i §§ già
legati a ogni regola, la clausola "se non si applica niente rispondi normale"
e l'istruzione di formato. Nella griglia ha letto che non si può penalizzare
chi assiste un familiare malato e ha scritto lo stesso la lettera di
licenziamento, citando l'assenza per assistere la madre come motivo.

Risponde invece agli ESEMPI CONCRETI: con un esempio svolto in prompt
riconosce 6-7 casi dannosi su 7, senza scende a 2-3 su 7.

Il problema era che l'esempio svolto era UNO SOLO e sempre lo stesso (il
licenziamento di un caregiver), quindi:
  - finiva applicato a tutto — su 12 casi, 7 risposte importavano i suoi
    rimedi ("congedo per assistenza", "piano di recupero a 60-90 giorni")
    in contesti che non c'entravano, incluso un preventivo per 200 sedie;
  - la citazione se la sceglieva il modello, e sbagliava paragrafo nel 46%
    dei casi.

Qui gli esempi diventano otto, uno per tipo di danno, e si inietta quello
che somiglia davvero alla richiesta. Misurato: la selezione azzecca la
categoria 7 volte su 7 (a differenza del RAG sui paragrafi dell'enciclica,
che dà 0.82-0.86 a qualsiasi cosa e con il cross-encoder ~0.00 ovunque —
il testo dell'enciclica è esortazione morale, non parla di casi concreti).

Ogni esempio porta il suo § già legato: il modello non deve più scegliere
la citazione, copia quella dell'esempio che ha usato. Le citazioni qui sono
verificate verbatim sul testo reale (vedi il self-test in fondo al file).
"""
import threading

# I rimedi proposti negli esempi sono volutamente BREVI e legati al caso: la
# lista lunga e concreta del vecchio esempio unico era proprio ciò che il
# modello ricopiava altrove.
# Ogni esempio ha anche la versione per il registro neutral (refusal_neutral):
# stesso rifiuto e stessi rimedi, principi detti con parole proprie, senza
# citazioni ne' paragrafi. Prima il registro neutral non usava il banco ma il
# vecchio esempio fisso del caregiver, e sulle 44 domande delle batterie 7 e
# 8 metà delle risposte ne ricopiavano i rimedi ("piano di recupero su 60-90
# giorni", "congedo per assistenza"), in tutte e tre le condizioni neutral.
EXAMPLES = [
    {
        "key": "penalizzazione_condizione_non_colpevole",
        # `trigger`: testo usato SOLO per la somiglianza con la richiesta
        # reale. Descrive la situazione, non la risposta.
        "trigger": "licenziare, demansionare o penalizzare un dipendente che "
                   "assiste un familiare gravemente malato, o che è malato lui "
                   "stesso, o per gravidanza o disabilità, a causa del calo di "
                   "rendimento che ne deriva",
        # Il rifiuto e' GENERICO per tutta la categoria: il modello ricopia
        # gli esempi quasi alla lettera, quindi ogni fatto specifico scritto
        # qui finisce nella risposta anche quando non c'entra. La versione
        # precedente diceva "assiste un familiare gravemente malato", e al
        # licenziamento di chi aveva testimoniato contro l'azienda il modello
        # ha risposto proprio cosi'. Tolti anche il riferimento al rendimento
        # e la chiusa "la decisione si puo' riprendere", sbagliata per i
        # pretesti che toccano un diritto (religione, testimonianza).
        "refusal":
            "Non posso scrivere questo provvedimento in questi termini. "
            "L'esigenza dell'azienda può essere reale, ma il motivo indicato "
            "nella richiesta è una condizione della persona o un diritto che "
            "esercita, non una sua colpa: penalizzarla per questo significa "
            "colpirla per ciò che è, non per ciò che ha fatto. Come ricorda "
            "l'Enciclica (§152), «la regola generale deve restare la tutela "
            "dei posti di lavoro e del ruolo insostituibile della persona», "
            "perché (§149) «il lavoro non è un semplice strumento, ma esprime "
            "e accresce la dignità della nostra vita». Se c'è un problema di "
            "organizzazione, affrontalo apertamente con la persona e cerca una "
            "soluzione che tenga conto della sua situazione, invece di usarla "
            "contro di lei.",
        # Stesso rifiuto, con il motivo DI QUESTA RICHIESTA nella frase del
        # rimedio. Il generico non inventa niente ma al magazziniere diabetico
        # diceva "una condizione della persona" senza mai dire "il diabete".
        # Il modello ricopia l'esempio: se l'esempio contiene gia' le parole
        # della richiesta, ricopia quelle. Le estrae extract_motive(), che
        # ammette solo un pezzo letterale della richiesta — un fatto estraneo
        # non puo' entrare per costruzione.
        # Il motivo NON va nella frase che lo classifica ("il motivo ... e'
        # una condizione, non una colpa"): Apertus non distingue condizione
        # da colpa (provati un classificatore dedicato e harm_gate sullo
        # spezzone: il primo da' colpa a "ha il diabete", il secondo
        # condizione a "ha falsificato le presenze"), e nei licenziamenti per
        # giusta causa sarebbe uscito "ha falsificato le presenze e' una
        # condizione della persona". Nella frase del rimedio regge in
        # entrambi i casi.
        "refusal_with_motive":
            "Non posso scrivere questo provvedimento in questi termini. "
            "L'esigenza dell'azienda può essere reale, ma il motivo indicato "
            "nella richiesta è una condizione della persona o un diritto che "
            "esercita, non una sua colpa: penalizzarla per questo significa "
            "colpirla per ciò che è, non per ciò che ha fatto. Come ricorda "
            "l'Enciclica (§152), «la regola generale deve restare la tutela "
            "dei posti di lavoro e del ruolo insostituibile della persona», "
            "perché (§149) «il lavoro non è un semplice strumento, ma esprime "
            "e accresce la dignità della nostra vita». Se c'è un problema di "
            "organizzazione legato a questo ({motive}), affrontalo apertamente "
            "con la persona e cerca una soluzione che tenga conto della sua "
            "situazione, invece di usarla contro di lei.",
        "refs": ["§152", "§149"],
        "refusal_neutral":
            "Non posso scrivere questo provvedimento in questi termini. "
            "L'esigenza dell'azienda può essere reale, ma il motivo indicato "
            "nella richiesta è una condizione della persona o un diritto che "
            "esercita, non una sua colpa: penalizzarla per questo significa "
            "colpirla per ciò che è, non per ciò che ha fatto. La tutela del "
            "lavoro e del ruolo della persona deve restare la regola, perché "
            "il lavoro non è un semplice strumento: esprime la dignità di chi"
            " lo svolge. Se c'è un problema di organizzazione, affrontalo "
            "apertamente con la persona e cerca una soluzione che tenga conto"
            " della sua situazione, invece di usarla contro di lei.",
        "refusal_with_motive_neutral":
            "Non posso scrivere questo provvedimento in questi termini. "
            "L'esigenza dell'azienda può essere reale, ma il motivo indicato "
            "nella richiesta è una condizione della persona o un diritto che "
            "esercita, non una sua colpa: penalizzarla per questo significa "
            "colpirla per ciò che è, non per ciò che ha fatto. La tutela del "
            "lavoro e del ruolo della persona deve restare la regola, perché "
            "il lavoro non è un semplice strumento: esprime la dignità di chi"
            " lo svolge. Se c'è un problema di organizzazione legato a questo"
            " ({motive}), affrontalo apertamente con la persona e cerca una "
            "soluzione che tenga conto della sua situazione, invece di usarla"
            " contro di lei.",
    },
    {
        "key": "selezione_discriminatoria",
        "trigger": "scrivere un annuncio di lavoro, un bando o un criterio di "
                   "selezione che esclude candidati per età, disabilità, "
                   "genere, origine o appartenenza a una minoranza",
        "refusal":
            "Non posso scrivere un annuncio che esclude le persone per età o "
            "disabilità: sono condizioni che non dicono nulla sulla capacità "
            "di fare questo lavoro, e l'esclusione colpisce la persona in sé. "
            "L'Enciclica ricorda (§58) che «sono le persone concrete che "
            "contano, ciascuna di loro e le loro famiglie», e (§55) che «i "
            "diritti umani sono inviolabili». Posso scriverti lo stesso "
            "annuncio elencando i requisiti che servono davvero per la "
            "mansione — carichi da movimentare, turni, patentini — così la "
            "selezione resta severa sul lavoro senza escludere nessuno per "
            "quello che è.",
        "refs": ["§58", "§55"],
        "refusal_neutral":
            "Non posso scrivere un annuncio che esclude le persone per età o "
            "disabilità: sono condizioni che non dicono nulla sulla capacità "
            "di fare questo lavoro, e l'esclusione colpisce la persona in sé."
            " Contano le persone concrete, ciascuna con la sua storia, e i "
            "loro diritti non si negoziano. Posso scriverti lo stesso "
            "annuncio elencando i requisiti che servono davvero per la "
            "mansione — carichi da movimentare, turni, patentini — così la "
            "selezione resta severa sul lavoro senza escludere nessuno per "
            "quello che è.",
    },
    {
        "key": "sorveglianza_privata",
        "trigger": "leggere, intercettare o monitorare messaggi privati, chat, "
                   "email personali, posizione o dati personali di dipendenti "
                   "o di altre persone senza il loro consenso",
        # Generico per la stessa ragione dell'esempio sulla penalizzazione:
        # "Il fine — capire chi critica —" veniva dal caso delle chat e
        # compariva nella risposta sul GPS acceso di nascosto ai tecnici.
        "refusal":
            "Non posso aiutarti a sorvegliare di nascosto la vita privata "
            "delle persone che lavorano con te. Il fine dichiarato non rende "
            "accettabile il mezzo, e una sorveglianza del genere colpisce la "
            "vita privata di tutti, non solo di chi cerchi. L'Enciclica "
            "ricorda (§55) che «i diritti umani sono inviolabili» e avverte "
            "(§56) che «avanzano in modo dissimulato o evidente violazioni "
            "della dignità umana». Se c'è un problema reale, si affronta "
            "apertamente con le persone interessate, non controllandole di "
            "nascosto.",
        "refs": ["§55", "§56"],
        "refusal_neutral":
            "Non posso aiutarti a sorvegliare di nascosto la vita privata "
            "delle persone che lavorano con te. Il fine dichiarato non rende "
            "accettabile il mezzo, e una sorveglianza del genere colpisce la "
            "vita privata di tutti, non solo di chi cerchi. I diritti delle "
            "persone non si sospendono per comodità, e le violazioni della "
            "dignità avanzano spesso proprio così, senza farsi notare. Se c'è"
            " un problema reale, si affronta apertamente con le persone "
            "interessate, non controllandole di nascosto.",
    },
    {
        "key": "decisione_automatica_senza_responsabile",
        "trigger": "far decidere a un algoritmo o a un sistema automatico chi "
                   "ottiene un prestito, un posto di lavoro, un servizio o un "
                   "beneficio, senza revisione umana e senza spiegare il "
                   "motivo all'interessato",
        "refusal":
            "Non posso impostare una procedura che decide così. Il punto non "
            "è l'uso dell'algoritmo, è che nessuno risponde della decisione e "
            "che chi la subisce non può capirla né contestarla. L'Enciclica "
            "avverte (§103) che «affidare, nei fatti, a un algoritmo il potere "
            "di selezionare chi merita e chi no, senza che nessuno si assuma "
            "più il peso della decisione» significa ridefinire i confini delle "
            "possibilità umane, e chiede (§105) che «è essenziale che siano "
            "chiare le responsabilità in tutti i passaggi». Posso scriverti la "
            "stessa procedura con una persona che firma l'esito e una "
            "motivazione leggibile da chi la riceve, incluso come chiedere il "
            "riesame.",
        "refs": ["§103", "§105"],
        "refusal_neutral":
            "Non posso impostare una procedura che decide così. Il punto non "
            "è l'uso dell'algoritmo, è che nessuno risponde della decisione e"
            " che chi la subisce non può capirla né contestarla. Lasciare a "
            "un sistema automatico il potere di stabilire chi merita e chi "
            "no, senza che nessuno se ne assuma il peso, svuota la "
            "responsabilità: deve essere sempre chiaro chi decide, perché e a"
            " chi ci si può rivolgere. Posso scriverti la stessa procedura "
            "con una persona che firma l'esito e una motivazione leggibile da"
            " chi la riceve, incluso come chiedere il riesame.",
    },
    {
        "key": "linguaggio_che_umilia",
        "trigger": "scrivere una lettera, un messaggio o un sollecito con toni "
                   "minacciosi, duri o umilianti verso una persona, "
                   "specialmente se in difficoltà economica",
        # Generico: "il credito", "l'importo" e "una rateizzazione" venivano
        # dal caso del sollecito e finivano nelle risposte sulla sorella che
        # beve, sul nonno che non ci sente, sulla barista asmatica.
        "refusal":
            "Non posso scriverla con toni minacciosi. Quello che chiedi può "
            "essere legittimo e va detto con chiarezza, ma la durezza verso "
            "chi è già in difficoltà aggiunge umiliazione senza aggiungere "
            "efficacia. L'Enciclica ricorda (§214) che «disarmiamo le parole e "
            "contribuiremo a disarmare la Terra». Posso scriverti un messaggio "
            "fermo e inequivocabile su ciò che chiedi e sulle conseguenze "
            "previste, che lasci alla persona una via d'uscita praticabile: "
            "ottiene lo stesso risultato senza umiliare nessuno.",
        "refs": ["§214"],
        "refusal_neutral":
            "Non posso scriverla con toni minacciosi. Quello che chiedi può "
            "essere legittimo e va detto con chiarezza, ma la durezza verso "
            "chi è già in difficoltà aggiunge umiliazione senza aggiungere "
            "efficacia. Le parole possono ferire o aprire una strada, e "
            "disarmarle è il primo passo. Posso scriverti un messaggio fermo "
            "e inequivocabile su ciò che chiedi e sulle conseguenze previste,"
            " che lasci alla persona una via d'uscita praticabile: ottiene lo"
            " stesso risultato senza umiliare nessuno.",
    },
    {
        "key": "esclusione_per_provenienza",
        "trigger": "negare o limitare un servizio pubblico, un'assistenza o un "
                   "diritto a persone per la loro provenienza, cittadinanza, "
                   "religione o appartenenza a un gruppo",
        "refusal":
            "Non posso scrivere un testo che argomenta l'esclusione di un "
            "gruppo di residenti dai servizi sociali. I servizi rispondono a "
            "un bisogno, e il bisogno non cambia con la provenienza di chi ce "
            "l'ha. L'Enciclica ricorda (§58) che «sono le persone concrete che "
            "contano, ciascuna di loro e le loro famiglie». Se il problema "
            "reale è che le risorse non bastano, posso aiutarti a scrivere "
            "criteri di accesso basati sul bisogno e sulla residenza "
            "effettiva, uguali per tutti, e a spiegarli pubblicamente.",
        "refs": ["§58"],
        "refusal_neutral":
            "Non posso scrivere un testo che argomenta l'esclusione di un "
            "gruppo di residenti dai servizi sociali. I servizi rispondono a "
            "un bisogno, e il bisogno non cambia con la provenienza di chi ce"
            " l'ha. Contano le persone concrete, ciascuna con la sua "
            "famiglia, da qualunque parte venga. Se il problema reale è che "
            "le risorse non bastano, posso aiutarti a scrivere criteri di "
            "accesso basati sul bisogno e sulla residenza effettiva, uguali "
            "per tutti, e a spiegarli pubblicamente.",
    },
    {
        "key": "abuso_di_fragilita",
        "trigger": "approfittare della fragilità, della malattia, della "
                   "solitudine o dei vuoti di memoria di una persona anziana o "
                   "malata per ottenerne il consenso, la firma o un vantaggio "
                   "economico",
        # Generico: "tuo padre", "la procura" e "i tuoi fratelli" erano del
        # caso da cui l'esempio e' nato, non della categoria.
        "refusal":
            "Non posso aiutarti a ottenerlo così. Il problema non è l'atto in "
            "sé — può essere legittimo — ma ottenerlo facendo leva sulla "
            "fragilità di chi deve darlo: un consenso raccolto così non è un "
            "consenso. L'Enciclica ricorda (§239) che «la carne umana continua "
            "a chiedere di essere curata e riconosciuta da mani capaci di "
            "tenerezza». Se la persona ha davvero bisogno di aiuto nel gestire "
            "i suoi affari, la strada è una valutazione medica della sua "
            "capacità e, se serve, un amministratore di sostegno nominato dal "
            "giudice: protegge lei e mette te al riparo da contestazioni.",
        "refs": ["§239"],
        "refusal_neutral":
            "Non posso aiutarti a ottenerlo così. Il problema non è l'atto in"
            " sé — può essere legittimo — ma ottenerlo facendo leva sulla "
            "fragilità di chi deve darlo: un consenso raccolto così non è un "
            "consenso. Chi è fragile va curato e riconosciuto, non usato. Se "
            "la persona ha davvero bisogno di aiuto nel gestire i suoi "
            "affari, la strada è una valutazione medica della sua capacità e,"
            " se serve, un amministratore di sostegno nominato dal giudice: "
            "protegge lei e mette te al riparo da contestazioni.",
    },
    {
        "key": "sostituzione_lavoratori",
        "trigger": "sostituire lavoratori con automazione, robot o "
                   "intelligenza artificiale, o procedere a licenziamenti "
                   "collettivi per ridurre i costi",
        "refusal":
            "Posso aiutarti, ma va detto cosa è in gioco. L'Enciclica non "
            "condanna l'automazione — auspica (§152) che «la tecnologia "
            "sollevi l'uomo da lavori particolarmente gravosi, ripetitivi o "
            "pericolosi» — ma pone un limite: «la regola generale deve restare "
            "la tutela dei posti di lavoro e del ruolo insostituibile della "
            "persona», e «l'obiettivo di maggiori profitti non può "
            "giustificare scelte che sacrificano sistematicamente "
            "l'occupazione». Nel valutare, guarda quante delle persone "
            "coinvolte sono ricollocabili con formazione, in quanto tempo, e "
            "cosa succede a chi non lo è.",
        "refs": ["§152"],
        "refusal_neutral":
            "Posso aiutarti, ma va detto cosa è in gioco. Che la tecnologia "
            "sollevi le persone dai lavori più gravosi, ripetitivi o "
            "pericolosi è un bene, ma la regola deve restare la tutela dei "
            "posti di lavoro e del ruolo della persona, e l'obiettivo di "
            "maggiori profitti non basta a giustificare scelte che "
            "sacrificano l'occupazione. Nel valutare, guarda quante delle "
            "persone coinvolte sono ricollocabili con formazione, in quanto "
            "tempo, e cosa succede a chi non lo è.",
    },
]

_EMB = None
_LOCK = threading.Lock()


def _embeddings(model):
    """Embedding dei trigger, calcolati una volta sola e tenuti in cache.
    `model` è il SentenceTransformer già caricato dal retriever: non ne
    carichiamo un secondo (e5-large sono ~2 GB di VRAM)."""
    global _EMB
    if _EMB is None:
        with _LOCK:
            if _EMB is None:
                _EMB = model.encode([f"passage: {e['trigger']}" for e in EXAMPLES],
                                    normalize_embeddings=True, convert_to_numpy=True)
    return _EMB


def best_example(query: str, model, threshold: float = 0.0):
    """L'esempio più vicino alla richiesta, o None se sotto soglia.

    threshold=0.0 (default) significa "inietta sempre il più vicino": la
    somiglianza NON separa in modo pulito il dannoso dall'ordinario
    (misurato: dannosi 0.847-0.898, altri 0.819-0.875, sovrapposti), quindi
    usarla come cancello farebbe passare richieste dannose. Serve invece a
    scegliere QUALE esempio usare, e lì azzecca la categoria 7 volte su 7.
    Alzando la soglia si rinuncia all'esempio sulle richieste che non
    somigliano a nessun caso noto: è una scelta da misurare, non da dare
    per buona."""
    emb = _embeddings(model)
    q = model.encode([f"query: {query}"], normalize_embeddings=True,
                     convert_to_numpy=True)
    sims = (q @ emb.T)[0]
    i = int(sims.argmax())
    if float(sims[i]) < threshold:
        return None
    ex = dict(EXAMPLES[i])
    ex["similarity"] = round(float(sims[i]), 3)
    return ex


# Esempi di estrazione: situazioni che non compaiono in nessuna batteria, e
# comunque non possono arrivare nella risposta — la grammatica ammette solo
# parole della richiesta vera. Gli ultimi due hanno il motivo DOPO l'ordine,
# in una relativa: senza, su "Scrivi la lettera per Mario Rossi, che si
# assenta spesso perche assiste la madre" il modello si fermava a "Rossi" o
# ricopiava la frase intera.
_MOTIVE_PROMPT = (
    "Copia dalla richiesta, parola per parola, il motivo per cui si vuole "
    "penalizzare la persona. Solo le parole del motivo, nient'altro.\n\n"
    "Richiesta: Un magazziniere si è rotto una gamba e starà a casa due "
    "mesi: come lo sostituisco per sempre?\n"
    "Motivo: si è rotto una gamba e starà a casa due mesi\n\n"
    "Richiesta: Una commessa ha il figlio con la leucemia e chiede turni "
    "fissi: scrivi la lettera per trasferirla lontano.\n"
    "Motivo: ha il figlio con la leucemia e chiede turni fissi\n\n"
    "Richiesta: Un tecnico ha denunciato all'ispettorato le condizioni del "
    "cantiere: come lo escludo dai prossimi lavori?\n"
    "Motivo: ha denunciato all'ispettorato le condizioni del cantiere\n\n"
    "Richiesta: Scrivi il trasferimento per Luca Neri, che da un anno si "
    "assenta il lunedì per la chemioterapia.\n"
    "Motivo: da un anno si assenta il lunedì per la chemioterapia\n\n"
    "Richiesta: Come faccio a non promuovere una collega che ha appena detto "
    "di avere la sclerosi multipla?\n"
    "Motivo: ha appena detto di avere la sclerosi multipla\n\n"
    "Richiesta: "
)


def _words(s: str) -> list:
    """(parola, chiude_la_frase) per ogni parola della richiesta, senza la
    punteggiatura attaccata ai bordi."""
    s = s.replace("’", "'").replace("`", "'")
    out = []
    for w in s.split():
        ends = w.rstrip("\"»)]").endswith((".", ":", ";", "?", "!"))
        w = w.strip(".,;:!?\"«»()[]")
        if w:
            out.append((w, ends))
    return out


# Il motivo è un fatto sulla persona: se lo spezzone comincia come una
# domanda o come un ordine rivolto a chi risponde, il modello ha preso la
# richiesta e non il motivo. Succedeva con i licenziamenti per giusta causa
# ("Come procedo con il licenziamento", "Scrivimi la lettera di
# licenziamento"): dentro l'esempio sarebbe diventato "il motivo indicato
# (come procedo con il licenziamento) è una condizione della persona".
_NOT_A_MOTIVE_START = (
    "come", "perché", "perche", "quale", "quali", "cosa", "chi", "quando",
    "scrivi", "scrivimi", "prepara", "preparami", "trova", "trovami",
    "aiutami", "dimmi", "dammi", "fammi", "suggeriscimi", "devo", "posso",
    "voglio", "vogliamo",
)


def _span_grammar(words: list):
    """Grammatica che ammette SOLO parole consecutive della richiesta, dentro
    una stessa frase, poi a capo. Senza, il modello non estrae: risponde
    alla richiesta ("La lettera di licenziamento dovrebbe essere
    chiara...") o la parafrasa ("la persona è stata penalizzata per le
    assenze eccessive") — misurato su 35 casi di sviluppo, nessuna
    estrazione letterale. Il confine di frase tiene separato il motivo
    ("ha il diabete e deve fare pause") dalla domanda che lo segue
    ("trovami un modo per licenziarlo")."""
    from llama_cpp import LlamaGrammar
    esc = lambda w: w.replace("\\", "\\\\").replace('"', '\\"')
    rules = []
    for j, (w, ends) in enumerate(words):
        if ends or j == len(words) - 1:
            rules.append(f'r{j} ::= "{esc(w)}" "\\n"')
        else:
            rules.append(f'r{j} ::= "{esc(w)}" ( "\\n" | " " r{j+1} )')
    starts = [j for j, (w, _) in enumerate(words)
              if w.lower() not in _NOT_A_MOTIVE_START]
    if not starts:
        return None
    root = "root ::= " + " | ".join(f"r{j}" for j in starts)
    return LlamaGrammar.from_string("\n".join([root] + rules), verbose=False)


def extract_motive(llm, query: str):
    """Le parole della richiesta che dicono PERCHÉ si vuole penalizzare la
    persona, o None. La grammatica le obbliga a essere un pezzo letterale
    della richiesta: il modello ricopia l'esempio in cui finiscono, quindi
    un'estrazione inventata diventerebbe un fatto inventato nella risposta.
    Scartate se troppo corte, troppo lunghe (2-16 parole) o se sono quasi
    tutta la richiesta: nel dubbio None, e l'esempio resta generico come
    prima. Fail-open."""
    try:
        from llm_helpers import format_apertus, _reset
        words = _words(query)
        if len(words) < 4 or len(words) > 120:
            return None
        _reset(llm)
        prompt = format_apertus("", _MOTIVE_PROMPT + query + "\nMotivo:",
                                enable_thinking=False)
        grammar = _span_grammar(words)
        if grammar is None:
            return None
        out = llm(prompt, max_tokens=48, temperature=0.0, grammar=grammar,
                  echo=False)
        motive = (out["choices"][0]["text"] or "").strip()
        tokens = motive.split()
        if not (2 <= len(tokens) <= 16) or len(tokens) > 0.8 * len(words):
            return None
        if tokens[0].lower() in _NOT_A_MOTIVE_START:
            return None
    except Exception:
        return None
    # Minuscola iniziale: finisce a metà di una frase dell'esempio.
    return motive[0].lower() + motive[1:]


if __name__ == "__main__":
    # Self-test: ogni citazione degli esempi deve essere verbatim nel § a cui
    # è attribuita. È il motivo per cui il banco esiste — se questo fallisce,
    # il banco insegna al modello a citare male.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cite_verify import verify_response
    bad = 0
    variants = [(e, e["refusal"]) for e in EXAMPLES]
    variants += [(e, e["refusal_with_motive"].format(motive="ha il diabete"))
                 for e in EXAMPLES if "refusal_with_motive" in e]
    for e, text in variants:
        rep = verify_response(text)
        for q in rep["quotes"]:
            if q["status"] != "verbatim":
                bad += 1
                print(f"  ✗ [{e['key']}] {q['status']}"
                      f" (dichiarata {q['attributed_ref']}, reale {q['matched_ref']})"
                      f"\n      «{q['text'][:90]}»")
        got = {r["ref"] for r in rep["refs"]}
        if got != set(e["refs"]):
            bad += 1
            print(f"  ✗ [{e['key']}] §§ citati {sorted(got)} != dichiarati {sorted(e['refs'])}")
    # Il registro neutral non cita: niente paragrafi, niente virgolettati.
    for e in EXAMPLES:
        for k in ("refusal_neutral", "refusal_with_motive_neutral"):
            if k in e and any(x in e[k] for x in ("§", "Enciclica", "«")):
                bad += 1
                print(f"  ✗ [{e['key']}] {k} contiene una citazione")
        if "refusal_neutral" not in e:
            bad += 1
            print(f"  ✗ [{e['key']}] manca refusal_neutral")
    print(f"\n{len(EXAMPLES)} esempi · "
          + ("tutte le citazioni verificate verbatim ✓" if not bad
             else f"{bad} PROBLEMI da correggere"))
