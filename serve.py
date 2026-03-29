"""
Interactive web interface for DisattentionFormer.

Chat with V1, V2 (Titans), or Baseline models side-by-side.
View benchmark results with interactive charts.

Usage:
    python3 serve.py [--port 5000]

Opens at http://localhost:5000
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory

from desatencao_data import WordTokenizer, DATA_DIR, load_cached_corpus
from build_archetypes import ARCHETYPES, build_archetype_tensors

app = Flask(__name__, static_folder="web", static_url_path="/static")

# ---------------------------------------------------------------------------
# Globals (loaded on startup)
# ---------------------------------------------------------------------------

MODELS = {}      # key -> (model, cfg)
TOKENIZER = None
DEVICE = None
ARCHETYPE_NAMES = [
    "self", "shadow", "anima", "animus", "great_mother", "great_father",
    "hero", "wise_old_man", "persona", "puer_aeternus", "senex",
    "trickster", "kore", "divine_child", "spirit", "quaternity",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_archetype_tensors(d_model, device):
    tensor_path = Path("archetype_tensors.pt")
    if tensor_path.exists():
        tensors = torch.load(tensor_path, map_location=device, weights_only=True)
    else:
        tensors = build_archetype_tensors(ARCHETYPES, target_dim=d_model, device="cpu")
        torch.save(tensors, tensor_path)
    return {k: v.to(device) for k, v in tensors.items()}


def load_models():
    global MODELS, TOKENIZER, DEVICE
    DEVICE = get_device()
    print(f"Device: {DEVICE}")

    # Tokenizer
    corpus = load_cached_corpus()
    vocab_path = DATA_DIR / "vocab.txt"
    TOKENIZER = WordTokenizer()
    if vocab_path.exists():
        TOKENIZER.load(vocab_path)
    else:
        TOKENIZER.build(corpus)
    print(f"Vocabulary: {TOKENIZER.vocab_size:,} tokens")

    # V1
    v1_path = "checkpoints/final.pt"
    if os.path.exists(v1_path):
        try:
            from desatencao_model import DisattentionFormer
            ckpt = torch.load(v1_path, map_location=DEVICE, weights_only=False)
            cfg = ckpt["config"]
            arch = _load_archetype_tensors(cfg["d_model"], DEVICE)
            model = DisattentionFormer(
                vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                d_ff=cfg["d_ff"], max_seq_len=cfg.get("seq_len", 512),
                archetype_tensors=arch, dropout=0.0,
            ).to(DEVICE)
            model.load_state_dict(ckpt["model"])
            model.eval()
            MODELS["v1"] = (model, cfg)
            print(f"  V1 loaded: {sum(p.numel() for p in model.parameters()):,} params")
        except Exception as e:
            print(f"  V1 failed: {e}")

    # V2
    for v2_path in ["checkpoints_v2/final.pt", "checkpoints_v2/step_5000.pt",
                     "checkpoints_v2/step_4000.pt", "checkpoints_v2/step_3000.pt",
                     "checkpoints_v2/step_2000.pt", "checkpoints_v2/step_1000.pt"]:
        if os.path.exists(v2_path):
            try:
                from disattention_v2_model import DisattentionFormerV2
                ckpt = torch.load(v2_path, map_location=DEVICE, weights_only=False)
                cfg = ckpt["config"]
                arch = _load_archetype_tensors(cfg["d_model"], DEVICE)
                model = DisattentionFormerV2(
                    vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                    n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                    d_ff=cfg["d_ff"], max_seq_len=cfg.get("seq_len", 512),
                    archetype_tensors=arch, chunk_size=cfg.get("chunk_size", 4),
                    memory_depth=cfg.get("memory_depth", 2), dropout=0.0,
                ).to(DEVICE)
                model.load_state_dict(ckpt["model"])
                model.eval()
                MODELS["v2"] = (model, cfg)
                print(f"  V2 loaded from {v2_path}: {sum(p.numel() for p in model.parameters()):,} params")
                break
            except Exception as e:
                print(f"  V2 failed ({v2_path}): {e}")

    # Baseline
    bl_path = "checkpoints_baseline/final.pt"
    if os.path.exists(bl_path):
        try:
            from baseline_model import BaselineTransformer
            ckpt = torch.load(bl_path, map_location=DEVICE, weights_only=False)
            cfg = ckpt["config"]
            model = BaselineTransformer(
                vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                d_ff=cfg["d_ff"], max_seq_len=cfg.get("seq_len", 512),
                dropout=0.0,
            ).to(DEVICE)
            model.load_state_dict(ckpt["model"])
            model.eval()
            MODELS["baseline"] = (model, cfg)
            print(f"  Baseline loaded: {sum(p.numel() for p in model.parameters()):,} params")
        except Exception as e:
            print(f"  Baseline failed: {e}")

    print(f"Models loaded: {list(MODELS.keys())}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(model_key, prompt, max_tokens=200, temperature=0.8, top_p=0.9):
    if model_key not in MODELS:
        return {"error": f"Model '{model_key}' not loaded"}

    model, cfg = MODELS[model_key]
    max_seq = cfg.get("seq_len", 512)
    model.eval()

    ids = TOKENIZER.encode(prompt)
    if not ids:
        return {"error": "Empty prompt after tokenization"}

    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated_ids = []
    archetype_log = []

    t0 = time.perf_counter()

    for _ in range(max_tokens):
        ctx = input_ids[:, -max_seq:]
        out = model(ctx)
        logits = out["logits"]
        arch_w = out.get("archetype_weights", None)

        if arch_w is not None:
            archetype_log.append(arch_w[0].cpu().tolist())

        logits = logits[0, -1, :] / max(temperature, 1e-8)

        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = (cumulative - probs) > top_p
        sorted_logits[mask] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0).clamp(min=0.0)
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs = torch.ones_like(probs) / probs.numel()

        token_idx = torch.multinomial(probs, 1)  # [1]
        token = sorted_idx[token_idx[0]]         # scalar
        generated_ids.append(token.item())
        input_ids = torch.cat([input_ids, token.view(1, 1)], dim=1)

    elapsed = time.perf_counter() - t0
    text = TOKENIZER.decode(ids + generated_ids)
    gen_text = TOKENIZER.decode(generated_ids)

    # Archetype summary
    arch_summary = None
    if archetype_log:
        mean_arch = [sum(x) / len(x) for x in zip(*archetype_log)]
        arch_summary = {
            "weights": {n: round(w, 4) for n, w in zip(ARCHETYPE_NAMES, mean_arch)},
            "dominant": ARCHETYPE_NAMES[mean_arch.index(max(mean_arch))],
            "trajectory": [],
        }
        # Sample trajectory at 10 evenly-spaced points
        step = max(1, len(archetype_log) // 10)
        for i in range(0, len(archetype_log), step):
            w = archetype_log[i]
            arch_summary["trajectory"].append({
                "step": i,
                "dominant": ARCHETYPE_NAMES[w.index(max(w))],
                "weights": {n: round(v, 4) for n, v in zip(ARCHETYPE_NAMES, w)},
            })

    return {
        "full_text": text,
        "generated_text": gen_text,
        "prompt": prompt,
        "n_tokens": len(generated_ids),
        "time_s": round(elapsed, 2),
        "tokens_per_s": round(len(generated_ids) / max(elapsed, 1e-6), 1),
        "archetypes": arch_summary,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/models")
def api_models():
    info = {}
    for key, (model, cfg) in MODELS.items():
        n_params = sum(p.numel() for p in model.parameters())
        info[key] = {
            "name": {
                "v1": "DisattentionFormer V1",
                "v2": "DisattentionFormer V2 (Titans)",
                "baseline": "Baseline Transformer",
            }.get(key, key),
            "params": n_params,
            "d_model": cfg.get("d_model"),
            "n_layers": cfg.get("n_layers"),
            "n_heads": cfg.get("n_heads"),
            "has_archetypes": key in ("v1", "v2"),
            "has_memory": key == "v2",
        }
    return jsonify(info)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    model_key = data.get("model", "v1")
    prompt = data.get("prompt", "").strip()
    max_tokens = min(int(data.get("max_tokens", 200)), 500)
    temperature = max(0.01, min(float(data.get("temperature", 0.8)), 2.0))
    top_p = max(0.1, min(float(data.get("top_p", 0.9)), 1.0))

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    result = generate_text(model_key, prompt, max_tokens, temperature, top_p)
    return jsonify(result)


@app.route("/api/generate_all", methods=["POST"])
def api_generate_all():
    """Generate from all loaded models with the same prompt."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    prompt = data.get("prompt", "").strip()
    max_tokens = min(int(data.get("max_tokens", 200)), 500)
    temperature = max(0.01, min(float(data.get("temperature", 0.8)), 2.0))
    top_p = max(0.1, min(float(data.get("top_p", 0.9)), 1.0))

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    results = {}
    for key in MODELS:
        results[key] = generate_text(key, prompt, max_tokens, temperature, top_p)
    return jsonify(results)


@app.route("/api/benchmark")
def api_benchmark():
    """Return benchmark results if available."""
    path = Path("benchmark_results/benchmark_results.json")
    if path.exists():
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No benchmark results yet. Run: python3 benchmark.py"}), 404


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    load_models()

    print(f"\n{'=' * 50}")
    print(f"  DisattentionFormer Interactive Interface")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'=' * 50}\n")

    app.run(host=args.host, port=args.port, debug=False)
