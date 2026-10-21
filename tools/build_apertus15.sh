#!/usr/bin/env bash
# Apertus v1.5 (8B o 70B) → GGUF Q4_K_M per llama.cpp, dai pesi ufficiali.
#
#   tools/build_apertus15.sh 8B
#   tools/build_apertus15.sh 70B
#
# Perche' si costruisce invece di scaricare: i pesi ufficiali (swiss-ai) sono
# multimodali e llama.cpp non li carica; qui se ne estrae la parte testuale
# (extract_text_model.py), che e' un Apertus normale, e la si converte. Nessun
# GGUF di v1.5 pronto e' stato usato: cosi' la fedelta' la controlliamo noi.
#
# Prima di lanciarlo:
#   * la venv del progetto attiva-bile in ./venv, con llama-cpp-python (serve a
#     quantizzare) e `pip install -r requirements-build.txt`;
#   * il repo swiss-ai/Apertus-v1.5-* e' "gated": accettare la licenza sulla
#     pagina del modello e fare `huggingface-cli login` (token dell'utente);
#   * spazio al picco, se non si cancella nulla fra i passi (pesi ufficiali +
#     parte testuale + bf16 + Q4_K_M): ~55 GB (8B) o ~470 GB (70B); a lavoro
#     finito si possono cancellare models/hf e il .gguf bf16;
#   * RAM: l'estrazione della 70B legge file da 50 GB mappati in memoria; con
#     32 GB e il Playground acceso veniva terminata. Spegnere l'app prima.
#
# Ogni passo si salta se il suo risultato c'e' gia': si puo' rilanciare dopo
# un'interruzione. Variabili: LLAMA_CPP_DIR (default tools/llama.cpp, clonato al
# commit usato per i file in produzione), LLM_BASE (default: il repo).
set -euo pipefail

SIZE="${1:-}"
case "$SIZE" in 8B|70B) ;; *) echo "usage: $0 8B|70B" >&2; exit 2 ;; esac
LOW="$(echo "$SIZE" | tr 'B' 'b')"

ROOT="${LLM_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
# shellcheck disable=SC1091
source venv/bin/activate
NV="$(echo "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib | tr ' ' ':')"
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Il commit di llama.cpp con cui sono stati convertiti i GGUF di oggi. La
# conversione e' la parte che cambia da una versione all'altra (nomi dei
# tensori, metadati): non sostituirlo senza rifare i controlli del passo 5.
LLAMA_CPP_COMMIT=a97cce86a8addeb9f40cba7a261c94b1f0c576cb
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/tools/llama.cpp}"

SRC="models/hf/Apertus-v1.5-$SIZE"
TXT="models/hf/Apertus-v1.5-$SIZE-text"
BF16="models/gguf/apertus-v1.5-$LOW-bf16.gguf"
Q4="models/gguf/apertus-v1.5-$LOW-q4km.gguf"
mkdir -p models/hf models/gguf

echo "[$(date +%T)] 0/5 llama.cpp (only the conversion script)"
if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" checkout -q "$LLAMA_CPP_COMMIT"

echo "[$(date +%T)] 1/5 downloading the official weights"
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("swiss-ai/Apertus-v1.5-$SIZE", local_dir="$SRC", max_workers=8,
    allow_patterns=["model-apertus-model-*.safetensors", "model.safetensors.index.json", "config.json",
                    "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "chat_template.jinja", "LICENSE.txt", "README.md"])
PY

echo "[$(date +%T)] 2/5 extracting the text part"
if [ -s "$TXT/model.safetensors.index.json" ]; then
  echo "   already done"
else
  python3 extract_text_model.py "$SRC" "$TXT"
fi

echo "[$(date +%T)] 3/5 converting to GGUF bf16"
if [ -s "$BF16.ok" ]; then
  echo "   already done"
else
  LOGF="$(mktemp)"
  PYTHONPATH="$LLAMA_CPP_DIR/gguf-py" \
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$TXT" --outtype bf16 --outfile "$BF16" \
    >"$LOGF" 2>&1 || true
  grep -iE "eos|successfully|error|Traceback|Killed" "$LOGF" || true
  grep -q "successfully" "$LOGF" || { echo "conversion failed, log: $LOGF" >&2; exit 1; }
  echo ok > "$BF16.ok"
fi

echo "[$(date +%T)] 4/5 quantizing to Q4_K_M"
if [ -s "$Q4.ok" ]; then
  echo "   already done"
else
  python3 - <<PY
import llama_cpp, os
p = llama_cpp.llama_model_quantize_default_params()
p.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M; p.nthread = os.cpu_count() or 8
rc = llama_cpp.llama_model_quantize(b"$BF16", b"$Q4", p)
assert rc == 0, rc
PY
  echo ok > "$Q4.ok"
fi

echo "[$(date +%T)] 5/5 checks"
python3 - <<PY
import sys; sys.path.insert(0, ".")
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from llm_helpers import format_apertus
m = Llama(model_path="$Q4", vocab_only=True, verbose=False); md = m.metadata
print({k: md.get(k) for k in ("general.architecture", "apertus.block_count", "tokenizer.ggml.eos_token_id", "apertus.vocab_size")})
out = Jinja2ChatFormatter(template=md["tokenizer.chat_template"], eos_token="<|assistant_end|>", bos_token="<s>")(
    messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "CIAO"}]).prompt
assert md.get("tokenizer.ggml.eos_token_id") == "68", "end-of-answer token is not 68"
assert out == "<s>" + format_apertus("SYS", "CIAO", False), "chat template differs from format_apertus"
print("checks OK")
PY
ls -la "$Q4"
echo "[$(date +%T)] DONE"
