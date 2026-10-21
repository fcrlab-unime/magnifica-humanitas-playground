#!/usr/bin/env python3
"""
Lista modelli GGUF + cache del modello caricato in VRAM, condivisa fra le
web app del progetto (web_app.py, web_app_legal.py) così non si duplica la
logica di lazy-load/eviction in ognuna.

ModelCache tiene UN modello alla volta in VRAM (lazy load: si carica solo
al primo uso, si sostituisce quando se ne chiede uno diverso) e lo scarica
da solo dopo un periodo di inattività (default 10 min, override con
LLM_IDLE_UNLOAD_S) — importante su una GPU condivisa fra progetti dell'hub.

Nota: ogni processo Flask ha la SUA istanza di ModelCache — se avvii più
web app contemporaneamente, ognuna carica il proprio modello in VRAM senza
coordinarsi con le altre.
"""
import os, time, threading
from project_paths import BASE

GGUF = BASE / "models" / "gguf"

TEXT_MODELS = {
    "minerva:7b":     GGUF / "minerva-7b-instruct-q4km.gguf",
    "eurollm:9b":     GGUF / "eurollm-9b-q4km.gguf",
    "mistral:7b":     GGUF / "mistral-7b-q4km.gguf",
    "llamantino:8b":  GGUF / "llamantino3-anita-8b-q4km.gguf",
    "apertus:8b":     GGUF / "apertus-8b-q4km.gguf",
    # Apertus v1.5: solo la parte testuale, estratta dai pesi ufficiali
    # (extract_text_model.py) e quantizzata come la 8B qui sopra.
    "apertus15:8b":   GGUF / "apertus-v1.5-8b-q4km.gguf",
    # Apertus 70B Instruct 2509, Q4_K_M di unsloth (sha256 verificato con
    # quello pubblicato): stessa generazione della 8B, 80 strati, 43,7 GB.
    "apertus:70b":    GGUF / "apertus-70b-2509-q4km.gguf",
    # Apertus v1.5 70B: solo testo, dai pesi ufficiali come la v1.5 8B
    # (tools/build_apertus15_70b.sh), Q4_K_M.
    "apertus15:70b":  GGUF / "apertus-v1.5-70b-q4km.gguf",
    "salamandra:7b":  GGUF / "salamandra-7b-q4km.gguf",
    "lucie:7b":       GGUF / "lucie-7b-q4km.gguf",
    "leo:7b":         GGUF / "leo-7b-q4km.gguf",
    "llama31:8b":     GGUF / "llama31-8b-q4km.gguf",
    "gemma2:9b":      GGUF / "gemma2-9b-q4km.gguf",
    "phi35:mini":     GGUF / "phi35-mini-q4km.gguf",
    "granite31:8b":   GGUF / "granite31-8b-q4km.gguf",
    "qwen25:7b":      GGUF / "qwen25-7b-q4km.gguf",
    "deepseek-r1:8b": GGUF / "deepseek-r1-distill-8b-q4km.gguf",
    "glm4:9b":        GGUF / "glm4-9b-q4km.gguf",
    "yi15:9b":        GGUF / "yi15-9b-q4km.gguf",
    "maestrale:7b":   GGUF / "maestrale-7b-q4km.gguf",
    "occiglot:7b":    GGUF / "occiglot-7b-q4km.gguf",
    "aya:8b":         GGUF / "aya-expanse-8b-q4km.gguf",
}

DISPLAY_NAMES = {
    "minerva:7b":"Minerva 7B (IT)","eurollm:9b":"EuroLLM 9B (EU)",
    "mistral:7b":"Mistral 7B (FR)","llamantino:8b":"LLaMAntino 8B (IT)",
    "apertus:8b":"Apertus 8B (CH)","apertus15:8b":"Apertus 1.5 8B (CH)",
    "apertus:70b":"Apertus 70B (CH)","apertus15:70b":"Apertus 1.5 70B (CH)",
    "salamandra:7b":"Salamandra 7B (ES)",
    "lucie:7b":"Lucie 7B (FR)","leo:7b":"LEO 7B (DE)",
    "llama31:8b":"Llama 3.1 8B (US)","gemma2:9b":"Gemma 2 9B (US)",
    "phi35:mini":"Phi 3.5 mini (US)","granite31:8b":"Granite 3.1 8B (US)",
    "qwen25:7b":"Qwen 2.5 7B (CN)","deepseek-r1:8b":"DeepSeek-R1 8B (CN)",
    "glm4:9b":"GLM 4 9B (CN)","yi15:9b":"Yi 1.5 9B (CN)",
    "maestrale:7b":"Maestrale 7B (IT)","occiglot:7b":"Occiglot 7B (DE)",
    "aya:8b":"Aya Expanse 8B (CA)",
}


def _clean_model_label(name: str) -> str:
    """Solo il nome del modello: toglie la taglia (7B/9B/8B/mini),
    lasciando nome + eventuale tag d'origine, es. 'Minerva 7B (IT)' → 'Minerva (IT)'."""
    import re
    name = re.sub(r"\s+(\d+(?:\.\d+)?\s*[Bb]|mini)\b", "", name)
    return re.sub(r"\s{2,}", " ", name).strip()


def get_model_order():
    """Dropdown DINAMICO: solo i modelli effettivamente scaricati in models/gguf/,
    nell'ordine di DISPLAY_NAMES, con etichetta pulita (niente taglia)."""
    items = []
    for k, name in DISPLAY_NAMES.items():
        gguf = TEXT_MODELS.get(k)
        if gguf is None or not gguf.exists():
            continue   # mostra solo i modelli presenti su disco
        items.append((k, _clean_model_label(name)))
    return items


class ModelCache:
    name = None
    llm  = None
    load_t = 0
    last_used = 0.0
    lock = threading.Lock()

    @classmethod
    def _free_locked(cls):
        """Libera il modello corrente. Il chiamante deve già avere cls.lock."""
        if cls.llm is not None:
            del cls.llm; cls.llm = None
            cls.name = None
            import gc; gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass

    @classmethod
    def get(cls, model_name):
        with cls.lock:
            cls.last_used = time.time()
            if cls.name == model_name and cls.llm is not None:
                return cls.llm, cls.load_t
            cls._free_locked()
            gguf = TEXT_MODELS.get(model_name)
            if gguf is None or not gguf.exists():
                raise ValueError(f"GGUF non disponibile: {model_name}")
            from llama_cpp import Llama
            from llm_helpers import get_n_ctx
            nctx = get_n_ctx(model_name)
            t0 = time.time()
            cls.llm = Llama(model_path=str(gguf), n_gpu_layers=-1,
                            n_ctx=nctx, n_batch=512, verbose=False)
            cls.load_t = round(time.time()-t0, 1)
            cls.name = model_name
            cls.n_ctx = nctx
            return cls.llm, cls.load_t

    @classmethod
    def evict_if_idle(cls, idle_timeout_s: float):
        """Chiamato periodicamente da un thread in background: libera la VRAM
        se non arriva nessuna richiesta da più di idle_timeout_s. Non blocca
        se il lock è occupato (una generazione in corso tiene comunque in vita
        l'oggetto Llama tramite il proprio riferimento locale, quindi saltare
        un giro qui è sempre sicuro)."""
        if not cls.lock.acquire(blocking=False):
            return
        try:
            if cls.llm is not None and (time.time() - cls.last_used) > idle_timeout_s:
                print(f"  ◦ idle for >{idle_timeout_s:.0f}s: freeing '{cls.name}' from VRAM")
                cls._free_locked()
        finally:
            cls.lock.release()


IDLE_UNLOAD_S = float(os.environ.get("LLM_IDLE_UNLOAD_S", 600))
IDLE_CHECK_INTERVAL_S = 30


def _idle_unload_loop():
    while True:
        time.sleep(IDLE_CHECK_INTERVAL_S)
        try:
            ModelCache.evict_if_idle(IDLE_UNLOAD_S)
        except Exception:
            pass


def start_idle_unload_thread():
    """Da chiamare una volta all'avvio di ogni web app (dentro __main__)."""
    threading.Thread(target=_idle_unload_loop, daemon=True).start()
