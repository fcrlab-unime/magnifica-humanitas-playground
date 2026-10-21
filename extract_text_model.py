#!/usr/bin/env python3
"""
Estrae il modello linguistico da Apertus v1.5 (multimodale:
Apertus1p5ForConditionalGeneration, con tokenizer per immagini e audio)
come un Apertus solo testo (ApertusForCausalLM), l'architettura che
llama.cpp sa convertire e caricare.

La parte testuale di v1.5 e' la stessa architettura di Apertus 2509 — 32
strati, attivazione xIELU, qk-norm, stessi nomi dei pesi con il prefisso
"model.language_model." invece di "model." — e gli stessi token speciali
della chat (61-68), con lo stesso formato di conversazione (verificato col
template ufficiale). Cambia il vocabolario: 266752 voci in ingresso, perche'
ai 131072 token di testo si aggiungono quelli di immagini e audio, mentre
l'uscita (lm_head) resta sui 131072 di testo. Qui si tengono solo quelli.

Usa i pesi ufficiali di swiss-ai, non le estrazioni della comunita': cosi'
la fedelta' dell'estrazione la controlliamo noi.

Usage:
    python extract_text_model.py models/hf/Apertus-v1.5-8B models/hf/Apertus-v1.5-8B-text
"""
import json, shutil, sys
from pathlib import Path

# Blocchi da 2 GB: il container del progetto ha 32 GB di RAM, e con la 70B
# (file di pesi da 50 GB, letti in memoria mappata) blocchi da 5 GB facevano
# terminare l'estrazione per memoria esaurita.
SHARD_BYTES = 2 * 1024**3
PREFIX = "model.language_model."


def text_config(src: Path) -> dict:
    full = json.loads((src / "config.json").read_text())
    tc = dict(full["text_config"])
    n_text = tc.pop("output_vocab_size")
    tc.pop("model_type", None)
    rope = tc.get("rope_parameters") or {}
    tc.update({
        "architectures": ["ApertusForCausalLM"],
        "model_type": "apertus",
        "vocab_size": n_text,
        "tie_word_embeddings": False,
        # anche nel formato classico, per chi non legge rope_parameters
        "rope_theta": rope.get("rope_theta"),
        "rope_scaling": {k: v for k, v in rope.items() if k != "rope_theta"} or None,
    })
    return tc


def extract_weights(src: Path, dst: Path, n_text: int):
    from safetensors import safe_open
    from safetensors.torch import save_file
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    # Solo i file con pesi del modello linguistico: quelli dei tokenizer di
    # immagini e audio non servono, e di solito non si scaricano nemmeno.
    shards = sorted({sh for k, sh in index.items()
                     if k.startswith(PREFIX) or k == "lm_head.weight"})
    out, cur, cur_bytes, weight_map, n = [], {}, 0, {}, 0

    def flush():
        nonlocal cur, cur_bytes, n
        if not cur:
            return
        n += 1
        name = f"model-{n:05d}.safetensors"
        save_file(cur, str(dst / name), metadata={"format": "pt"})
        for k in cur:
            weight_map[k] = name
        cur, cur_bytes = {}, 0

    for shard in shards:
        with safe_open(str(src / shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith(PREFIX):
                    new = "model." + key[len(PREFIX):]
                elif key == "lm_head.weight":
                    new = key
                else:
                    continue  # tokenizer di immagini e audio
                t = f.get_tensor(key)
                if new == "model.embed_tokens.weight":
                    t = t[:n_text].contiguous()
                if new == "lm_head.weight" and t.shape[0] != n_text:
                    raise SystemExit(f"lm_head has {t.shape[0]} rows, expected {n_text}")
                cur[new] = t
                cur_bytes += t.numel() * t.element_size()
                if cur_bytes >= SHARD_BYTES:
                    flush()
    flush()
    total = sum((dst / s).stat().st_size for s in set(weight_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    return len(weight_map)


def extract_tokenizer(src: Path, dst: Path, n_text: int):
    tok = json.loads((src / "tokenizer.json").read_text())
    before = len(tok["added_tokens"])
    tok["added_tokens"] = [a for a in tok["added_tokens"] if a["id"] < n_text]
    (dst / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    if "added_tokens_decoder" in cfg:
        cfg["added_tokens_decoder"] = {k: v for k, v in cfg["added_tokens_decoder"].items()
                                       if int(k) < n_text}
    # Solo le chiavi di un tokenizer di testo: il resto (image_token,
    # audio_token, extra_special_tokens, omnimodal_config, …) rimette nel
    # vocabolario i token di immagini e audio appena tolti.
    keep = ("add_prefix_space", "bos_token", "eos_token", "pad_token",
            "unk_token", "clean_up_tokenization_spaces", "model_max_length",
            "model_input_names", "padding_side", "tokenizer_class",
            "added_tokens_decoder")
    cfg = {k: v for k, v in cfg.items() if k in keep}
    # Fine della risposta: in chat e' <|assistant_end|> (68). v1.5 dichiara
    # "</s>" nel tokenizer (68 e' solo in generation_config), e convertito
    # cosi' llama.cpp non si fermerebbe a fine risposta. Apertus 2509
    # convertito ha eos = 68.
    cfg["eos_token"] = "<|assistant_end|>"
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    stm = json.loads((src / "special_tokens_map.json").read_text())
    if isinstance(stm.get("eos_token"), dict):
        stm["eos_token"]["content"] = "<|assistant_end|>"
    else:
        stm["eos_token"] = "<|assistant_end|>"
    (dst / "special_tokens_map.json").write_text(json.dumps(stm, indent=1))
    for name in ("generation_config.json",
                 "chat_template.jinja", "LICENSE.txt", "README.md"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    return before - len(tok["added_tokens"])


def main(src: str, dst: str):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = text_config(src)
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    n_text = cfg["vocab_size"]
    n = extract_weights(src, dst, n_text)
    dropped = extract_tokenizer(src, dst, n_text)
    print(f"  ✓ {n} language-model tensors, vocabulary {n_text}, "
          f"{dropped} image and audio tokens removed from the tokenizer → {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python extract_text_model.py <source dir> <destination dir>")
    main(sys.argv[1], sys.argv[2])
