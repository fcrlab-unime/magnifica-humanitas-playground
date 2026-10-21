#!/usr/bin/env python3
"""
Path del progetto, risolti da variabile d'ambiente per la portabilità.

Su questa macchina i default funzionano senza configurare nulla. Su un altro
server basta esportare:

    export LLM_BASE=/percorso/del/repo
    export LLM_CUDA_LIB=/percorso/alle/librerie/cuda   # opzionale, per la GPU

Tutti i moduli core importano BASE da qui, invece di hard-codare il path.
"""
import os
import pathlib

# Cartella radice del progetto. Default: la cartella che contiene questo file
# (così funziona "out of the box" ovunque sia clonato il repo); override con LLM_BASE.
BASE = pathlib.Path(os.environ.get("LLM_BASE", str(pathlib.Path(__file__).resolve().parent)))

# Librerie CUDA extra per llama.cpp. Di norma non servono: gli script webapp*.sh
# usano quelle che torch installa nella venv (nvidia/*/lib). Si imposta
# LLM_CUDA_LIB solo se le librerie stanno altrove; viene messa davanti.
CUDA_LIB = os.environ.get("LLM_CUDA_LIB", "")
