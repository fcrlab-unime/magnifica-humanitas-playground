# Magnifica Humanitas Playground

A local web app that runs [Apertus](https://huggingface.co/swiss-ai) v1.5 (8B or 70B)
with [llama.cpp](https://github.com/ggml-org/llama.cpp) and answers in two modes:

- **Principles** — answers following the encyclical *Magnifica Humanitas*.
- **Documents** — retrieval over the GDPR, the AI Act (articles, annexes, recitals)
  and CEI documents, with checked citations and a link that opens the full source.

Requests are screened by a policy gate (`gate_policy.md`) before the model answers.

## Requirements

- Linux, NVIDIA GPU with a recent driver (memory needed: see [Choose a model](#choose-a-model))
- Python 3.13
- CUDA 12.x toolkit (`nvcc`), only to compile llama-cpp-python
- A Hugging Face account that has accepted the Apertus v1.5 license

## Install

```bash
./setup.sh install        # venv + pinned requirements
./setup.sh llama-cuda     # build llama-cpp-python with CUDA (without it, the app runs on CPU)

# Apertus v1.5 GGUF files are built from the official weights, not downloaded
source venv/bin/activate
pip install -r requirements-build.txt
huggingface-cli login
tools/build_apertus15.sh 8B        # or 70B: build only the one your GPU can hold

./setup.sh index          # build the retrieval indexes
./setup.sh check          # must end with "ALL READY"
```

The official v1.5 weights are multimodal and gated, so `extract_text_model.py`
keeps the text part and `tools/build_apertus15.sh` converts it to GGUF
(llama.cpp is pinned to the commit used in development).
Disk needed while building: ~55 GB (8B) or ~470 GB (70B); afterwards `models/hf` and the
bf16 `.gguf` can be deleted. Build the 70B with the app stopped.

## Choose a model

Build one size, or both: the app offers whatever is in `models/gguf/`, and opens with the
8B if present, otherwise with the one that is.

| | Build | File | GPU memory* |
|---|---|---|---|
| 8B (faster, less reliable) | `tools/build_apertus15.sh 8B` | 4.8 GB | ~13 GB |
| 70B (more reliable, slower) | `tools/build_apertus15.sh 70B` | 41 GB | ~54 GB |

\*Measured with the 16k context the app uses, including the search models (~5 GB).
All layers are loaded on the GPU: if the model does not fit, loading fails
(there is no CPU or partial offload).

## Run

```bash
./webapp_apertus.sh start      # http://localhost:5002
./webapp_apertus.sh status | logs | restart | stop
```

In the page: choose the model, switch between Principles and Documents
(EU law or CEI documents), pick a suggested question or type your own.
The *Chat memory* toggle makes follow-up questions use the previous turns;
*New chat* clears them. Only one model is kept in GPU memory at a time and it is
unloaded after 10 idle minutes.

## Configuration

All optional.

| Variable | Default | Meaning |
|---|---|---|
| `APERTUS_MODEL` | `apertus15:8b` | model selected when the page opens |
| `APERTUS_GATE` | `policy` | `policy` or `examples` (older gate) |
| `LLM_IDLE_UNLOAD_S` | `600` | idle seconds before the model leaves GPU memory |
| `LLM_BASE` | repo folder | project root |
| `LLM_CUDA_LIB` | — | extra CUDA library path |

The port (5002) is set in `web_app_apertus.py` and `webapp_apertus.sh`.

## Layout

| Path | |
|---|---|
| `web_app_apertus.py` | the app |
| `harm_gate.py`, `gate_policy.md` | policy gate |
| `docs_rag.py`, `rag_multisource.py`, `doc_view.py` | retrieval and Documents mode |
| `chat_memory.py` | follow-up questions |
| `model_cache.py`, `llm_helpers.py` | model loading and prompt format |
| `rag_encyclical.py`, `example_bank.py`, `query_expand.py`, `cite_verify.py` | encyclical index, suggested questions, query expansion, citation check |
| `docs_sample/` | the texts the app answers from: GDPR, AI Act, CEI documents, the encyclical |
| `setup.sh`, `tools/`, `extract_text_model.py` | installation and GGUF build |

Inference is llama.cpp throughout; vLLM is not used.
