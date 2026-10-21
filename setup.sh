#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Messa in opera del Playground su una macchina nuova.
#
#    ./setup.sh install      venv + requirements.txt (scarica ~10 GB di pacchetti)
#    ./setup.sh llama-cuda   ricompila llama-cpp-python CON CUDA (serve nvcc)
#    ./setup.sh index        costruisce gli indici RAG (ChromaDB) dai testi del repo
#    ./setup.sh check        verifica tutto e dice cosa manca; non installa niente
#
#  Ordine per una macchina vuota: install → llama-cuda → (modelli, vedi
#  README) → index → check → ./webapp_apertus.sh start
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
ROOT="${LLM_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT" || exit 1
PY="$ROOT/venv/bin/python"

nv_libs() {   # librerie CUDA che torch installa nella venv
    compgen -G "$ROOT/venv/lib/python*/site-packages/nvidia/*/lib" >/dev/null \
        && echo "$ROOT"/venv/lib/python*/site-packages/nvidia/*/lib | tr ' ' ':'
}
export LD_LIBRARY_PATH="${LLM_CUDA_LIB:+$LLM_CUDA_LIB:}$(nv_libs)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmd="${1:-check}"
case "$cmd" in
install)
    python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)' \
        || echo "WARNING: python3 is not 3.13; the pinned versions are only tested with 3.13"
    [ -x "$PY" ] || python3 -m venv venv || exit 1
    "$PY" -m pip install -q -U pip && "$PY" -m pip install -r requirements.txt || exit 1
    echo "✓ venv ready. Note: the llama-cpp-python from PyPI is a CPU build → ./setup.sh llama-cuda"
    ;;

llama-cuda)
    command -v nvcc >/dev/null 2>&1 || [ -n "${CUDACXX:-}" ] || {
        echo "nvcc not found: install the CUDA 12.x toolkit, or export CUDACXX=/path/to/nvcc"
        echo "(with conda: conda install -c nvidia 'cuda-toolkit>=12.4,<13', then CUDACXX=\$CONDA_PREFIX/bin/nvcc)"
        exit 1; }
    # architettura della scheda: 80 = A100, 89 = RTX 4090/L4, 90 = H100…
    ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')}"
    [ -n "$ARCH" ] || { echo "GPU architecture not detected: export CUDA_ARCH (e.g. CUDA_ARCH=80)"; exit 1; }
    VER="$(grep -E '^llama-cpp-python==' requirements.txt | cut -d= -f3)"
    echo "building llama-cpp-python==$VER for sm_$ARCH (~15 min)…"
    CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH" \
        "$PY" -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python \
        "llama-cpp-python==$VER" || exit 1
    ls "$ROOT"/venv/lib/python*/site-packages/llama_cpp/lib/libggml-cuda.so >/dev/null \
        && echo "✓ CUDA build present" || { echo "✗ libggml-cuda.so is missing: the build is still CPU-only"; exit 1; }
    ;;

index)
    "$PY" rag_encyclical.py index && "$PY" rag_multisource.py index
    ;;

check)
    bad=0
    "$PY" - <<'PY' || bad=1
import sys, glob, os
ok = True
def say(good, msg):
    global ok
    print(("  ✓ " if good else "  ✗ ") + msg); ok = ok and good
print(f"Python {sys.version.split()[0]}")
try:
    import llama_cpp
    cuda = bool(glob.glob(os.path.join(os.path.dirname(llama_cpp.__file__), "lib", "libggml-cuda.so")))
    say(True, f"llama-cpp-python {llama_cpp.__version__}")
    say(cuda, "CUDA build (libggml-cuda.so)" if cuda else "CPU-only build: run ./setup.sh llama-cuda")
except Exception as e:
    say(False, f"llama_cpp cannot be imported: {e}")
try:
    import torch
    say(torch.cuda.is_available(), "torch sees the GPU: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO"))
except Exception as e:
    say(False, f"torch cannot be imported: {e}")
base = os.environ.get("LLM_BASE") or os.getcwd()
sys.path.insert(0, base)
try:
    from model_cache import TEXT_MODELS
    found = [k for k in ("apertus15:8b", "apertus15:70b") if TEXT_MODELS[k].exists()]
    say(bool(found), "GGUF: " + (", ".join(found) if found else "no Apertus v1.5 in models/gguf/ (see README: tools/build_apertus15.sh)"))
except Exception as e:
    say(False, f"model_cache: {e}")
try:
    import chromadb
    names = {c.name for c in chromadb.PersistentClient(path="docs_sample/enciclica/chroma_db").list_collections()}
    need = {"multisource", "magnifica_humanitas"}
    say(need <= names, "RAG indexes: " + ("ok" if need <= names else f"missing {sorted(need - names)} → ./setup.sh index"))
except Exception as e:
    say(False, f"chromadb: {e}")
for p in ("docs_sample/corpus/gdpr/articles.jsonl", "docs_sample/corpus/ai_act/articles.jsonl",
          "docs_sample/corpus/cei_documenti/articles.jsonl", "docs_sample/enciclica/chunks.jsonl"):
    say(os.path.exists(os.path.join(base, p)), p)
print("\nALL READY" if ok else "\nNOT READY: see the ✗ lines")
sys.exit(0 if ok else 1)
PY
    exit $bad
    ;;

*)
    echo "usage: $0 {install|llama-cuda|index|check}"; exit 2 ;;
esac
