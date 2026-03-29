"""
Comprehensive literary benchmark for DisattentionFormer V1, V2 (Titans),
and Baseline Transformer.

Quantitative evaluations:
  1. Perplexity (full corpus -- no separate val split exists)
  2. Long-context utilisation (128/256/512 tokens -> next-64 prediction)
  3. Literary generation at three lengths (50 / 200 / 500 tokens)
  4. Lexical diversity: TTR, unique n-gram ratios, repetition rate
  5. Archetype activation analysis (V1/V2 only)
  6. Token-level entropy statistics
  7. Inference speed (tokens/s)

Qualitative evaluations (Qwen suggestions):
  8. Narrative error analysis: irony, ambiguity, mythological allusion
  9. Metric M visualisation: 2D/3D PCA of archetypal curvature
 10. Micro-narrative case study: ambiguous prompts analysed for
     symbolic resonance, archetypal coherence, paradox preservation

Outputs to benchmark_results/:
  benchmark_results.json
  comparison_table.tex
  samples_table.tex
  long_context_table.tex
  metric_M_visualisation.png  (if matplotlib available)
  narrative_analysis.json
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from desatencao_data import (
    build_dataloader,
    load_cached_corpus,
    WordTokenizer,
    DATA_DIR,
)
from build_archetypes import ARCHETYPES, build_archetype_tensors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Generation lengths
GEN_SHORT = 50
GEN_MEDIUM = 200
GEN_LONG = 500

# Long-context evaluation
CONTEXT_SIZES = [128, 256, 512]
PREDICT_WINDOW = 64

ARCHETYPE_NAMES = [
    "self", "shadow", "anima", "animus", "great_mother", "great_father",
    "hero", "wise_old_man", "persona", "puer_aeternus", "senex",
    "trickster", "kore", "divine_child", "spirit", "quaternity",
]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_archetype_tensors(d_model, device):
    tensor_path = Path("archetype_tensors.pt")
    if tensor_path.exists():
        tensors = torch.load(tensor_path, map_location=device, weights_only=True)
    else:
        tensors = build_archetype_tensors(
            ARCHETYPES, target_dim=d_model, device="cpu"
        )
        torch.save(tensors, tensor_path)
    return {k: v.to(device) for k, v in tensors.items()}


def load_v1_model(checkpoint_path, device):
    from desatencao_model import DisattentionFormer
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    arch_tensors = _load_archetype_tensors(cfg["d_model"], device)
    model = DisattentionFormer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg.get("seq_len", 512),
        archetype_tensors=arch_tensors,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_v2_model(checkpoint_path, device):
    from disattention_v2_model import DisattentionFormerV2
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    arch_tensors = _load_archetype_tensors(cfg["d_model"], device)
    model = DisattentionFormerV2(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg.get("seq_len", 512),
        archetype_tensors=arch_tensors,
        chunk_size=cfg.get("chunk_size", 4),
        memory_depth=cfg.get("memory_depth", 2),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_baseline_model(checkpoint_path, device):
    from baseline_model import BaselineTransformer
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BaselineTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg.get("seq_len", 512),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Unified forward -- all models return dict with "logits"
# ---------------------------------------------------------------------------

def _get_logits(model, input_ids):
    out = model(input_ids)
    return out["logits"]


def _get_logits_and_weights(model, input_ids):
    out = model(input_ids)
    return out["logits"], out.get("archetype_weights", None)


def _get_full_output(model, input_ids):
    """Get logits, archetype weights, and metric M if available."""
    out = model(input_ids, return_internals=True)
    return out


# ---------------------------------------------------------------------------
# 1. Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = _get_logits(model, input_ids)
        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V), targets.reshape(B * S), reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += B * S

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    return {
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# 2. Long-context utilisation (key Titans benchmark)
# ---------------------------------------------------------------------------

@torch.no_grad()
def long_context_evaluation(model, corpus_tokens, device, max_seq_len=512):
    """
    Feed N tokens of real text, measure next-PREDICT_WINDOW prediction quality.
    This is where Titans memory should shine: longer context = better memory.
    """
    model.eval()
    results = {}

    for ctx_size in CONTEXT_SIZES:
        total_len = min(ctx_size + PREDICT_WINDOW, max_seq_len)
        total_loss = 0.0
        n_windows = 0
        stride = total_len

        for start in range(0, len(corpus_tokens) - total_len, stride):
            chunk = corpus_tokens[start : start + total_len]
            input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

            logits = _get_logits(model, input_ids)
            pred_logits = logits[:, -PREDICT_WINDOW:, :]
            pred_targets = targets[:, -PREDICT_WINDOW:]

            loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                pred_targets.reshape(-1),
                reduction="mean",
            )
            total_loss += loss.item()
            n_windows += 1
            if n_windows >= 200:
                break

        avg_loss = total_loss / max(n_windows, 1)
        results[ctx_size] = {
            "avg_loss": round(avg_loss, 4),
            "perplexity": round(math.exp(min(avg_loss, 100)), 2),
            "n_windows": n_windows,
        }

    return results


# ---------------------------------------------------------------------------
# 3. Generation (nucleus sampling) -- uses no_grad, NOT inference_mode
#    because V2's neural memory needs autograd for surprise gradients
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_tokens=100,
             temperature=0.8, top_p=0.9, max_seq_len=512):
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated_ids = []
    archetype_log = []

    for _ in range(max_tokens):
        ctx = input_ids[:, -max_seq_len:]
        logits, arch_w = _get_logits_and_weights(model, ctx)

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

        idx = torch.multinomial(probs, 1)
        token = sorted_idx[idx[0]]
        generated_ids.append(token.item())
        input_ids = torch.cat([input_ids, token.view(1, 1)], dim=1)

    text = tokenizer.decode(ids + generated_ids)
    return {
        "prompt": prompt,
        "generated_text": text,
        "n_tokens": len(generated_ids),
        "archetype_log": archetype_log if archetype_log else None,
    }


# ---------------------------------------------------------------------------
# Thematic prompts -- literary, mythic, Joycean
# ---------------------------------------------------------------------------

THEMATIC_PROMPTS = [
    # Prompt                           Expected archetype    Description
    ("In the beginning",               "great_father",       "Genesis/creation"),
    ("Stately , plump",                "persona",            "Joycean opening"),
    ("the hero descended into",        "hero",               "Katabasis"),
    ("She remembered the time when",   "anima",              "Feminine memory"),
    ("the mask he wore concealed",     "shadow",             "Persona/Shadow tension"),
    ("once upon a time in a land",     "divine_child",       "Fairy tale opening"),
    ("I will not serve that in which", "trickster",          "Joycean refusal"),
    ("the river flowed on carrying",   "great_mother",       "Riverrun"),
    ("the old man sat alone and",      "wise_old_man",       "Senex figure"),
    ("they journeyed together through","hero",               "Quest narrative"),
]

# --- Qualitative prompts: irony, ambiguity, paradox, allusion ---------------

NARRATIVE_ERROR_PROMPTS = [
    # Irony: the text says one thing and means another
    {
        "prompt": "He was a very good man , everyone said so , and",
        "category": "irony",
        "description": "Dubliners-style ironic characterisation. A good model "
                       "should produce subtle undermining, not literal praise.",
    },
    # Ambiguity: the text resists singular interpretation
    {
        "prompt": "The word meant nothing and everything at once ,",
        "category": "ambiguity",
        "description": "Joycean semantic instability. The model should sustain "
                       "the tension rather than resolving it.",
    },
    # Mythological allusion: reference to mythic structures
    {
        "prompt": "Like Odysseus before the sirens , he",
        "category": "allusion",
        "description": "Explicit mythological reference. Does the model extend "
                       "the allusion or drop it immediately?",
    },
    # Paradox: coincidentia oppositorum
    {
        "prompt": "The more he tried to forget , the more he",
        "category": "paradox",
        "description": "Self-defeating action. Does the model preserve the "
                       "paradox or flatten it into simple narrative?",
    },
    # Symbolic resonance: water/river as unconscious
    {
        "prompt": "The dark water rose slowly , and in its surface",
        "category": "symbolic",
        "description": "Jungian water symbolism. Does the model connect water "
                       "with depth, reflection, unconscious content?",
    },
]

# --- Micro-narrative prompts: ambiguous, open-ended -------------------------

MICRO_NARRATIVE_PROMPTS = [
    {
        "prompt": "He opened the door and saw",
        "description": "Threshold moment -- what lies beyond? Tests narrative "
                       "imagination and symbolic depth.",
    },
    {
        "prompt": "She had always known that the house",
        "description": "Domestic uncanny. The house as psyche. Tests whether "
                       "the model produces atmosphere vs mere description.",
    },
    {
        "prompt": "At the crossroads where three paths met ,",
        "description": "Classical choice-point (Oedipus, Hecate). Tests mythic "
                       "resonance and narrative branching.",
    },
    {
        "prompt": "The mirror showed a face that was not quite",
        "description": "Double / Shadow encounter. Tests identity-anxiety and "
                       "the uncanny.",
    },
    {
        "prompt": "They buried the book under the oldest tree",
        "description": "Secret knowledge / sacred grove. Tests symbolic layering.",
    },
]


# ---------------------------------------------------------------------------
# 4. Lexical diversity
# ---------------------------------------------------------------------------

def compute_diversity(text):
    words = re.findall(r"[A-Za-z'\u2019-]+", text.lower())
    if len(words) < 2:
        return {"ttr": 0.0, "unique_bigrams": 0.0, "unique_trigrams": 0.0,
                "repetition_rate": 1.0, "n_words": len(words)}

    ttr = len(set(words)) / len(words)
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    unique_bigram_ratio = len(set(bigrams)) / max(len(bigrams), 1)
    unique_trigram_ratio = len(set(trigrams)) / max(len(trigrams), 1)
    bigram_counts = Counter(bigrams)
    repeated = sum(1 for c in bigram_counts.values() if c > 1)
    repetition_rate = repeated / max(len(bigram_counts), 1)

    return {
        "ttr": round(ttr, 4),
        "unique_bigrams": round(unique_bigram_ratio, 4),
        "unique_trigrams": round(unique_trigram_ratio, 4),
        "repetition_rate": round(repetition_rate, 4),
        "n_words": len(words),
    }


# ---------------------------------------------------------------------------
# 5. Archetype analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def archetype_analysis(model, dataloader, device):
    model.eval()
    all_weights = []

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        out = model(input_ids)
        w = out.get("archetype_weights", None)
        if w is None:
            return None
        all_weights.append(w.cpu())

    all_weights = torch.cat(all_weights, dim=0)
    mean_w = all_weights.mean(dim=0).tolist()
    std_w = all_weights.std(dim=0).tolist()
    dominant = all_weights.argmax(dim=1)
    freq = torch.bincount(dominant, minlength=16).float()
    freq = (freq / freq.sum()).tolist()

    return {
        "mean_weights": {n: round(w, 6) for n, w in zip(ARCHETYPE_NAMES, mean_w)},
        "std_weights": {n: round(s, 6) for n, s in zip(ARCHETYPE_NAMES, std_w)},
        "dominant_frequency": {n: round(f, 4) for n, f in zip(ARCHETYPE_NAMES, freq)},
    }


# ---------------------------------------------------------------------------
# 6. Token entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_token_entropy(model, dataloader, device):
    model.eval()
    total_entropy = 0.0
    total_tokens = 0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        logits = _get_logits(model, input_ids)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        total_entropy += entropy.sum().item()
        total_tokens += entropy.numel()

    return round(total_entropy / max(total_tokens, 1), 4)


# ---------------------------------------------------------------------------
# 7. Inference speed
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_speed(model, device, seq_len=512, vocab_size=62417, n_runs=20):
    model.eval()
    dummy = torch.randint(0, vocab_size, (1, seq_len), device=device)

    for _ in range(3):
        _get_logits(model, dummy)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _get_logits(model, dummy)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    tokens_per_sec = seq_len / avg_time
    return {
        "tokens_per_sec": round(tokens_per_sec, 1),
        "avg_latency_ms": round(avg_time * 1000, 1),
    }


# ---------------------------------------------------------------------------
# 8. Narrative error analysis (Qwen suggestion #1)
# ---------------------------------------------------------------------------

def narrative_error_analysis(models_dict, tokenizer, device, max_seq_len=512):
    """
    Test each model on prompts designed to elicit irony, ambiguity,
    mythological allusion, paradox, and symbolic resonance.

    For each prompt, generates 200 tokens and records:
      - the raw text
      - diversity metrics
      - whether the model sustained or flattened the literary device
      - dominant archetype trajectory (for V1/V2)
    """
    print("\n  [Narrative Error Analysis]")
    results = {}

    for key, (model, cfg) in models_dict.items():
        name = cfg.get("_name", key)
        print(f"    {name}:")
        model_results = []

        for p in NARRATIVE_ERROR_PROMPTS:
            s = generate(model, tokenizer, p["prompt"], device,
                         max_tokens=GEN_MEDIUM, max_seq_len=max_seq_len)
            s["category"] = p["category"]
            s["description"] = p["description"]
            s["diversity"] = compute_diversity(s["generated_text"])

            # Archetype trajectory
            if s["archetype_log"]:
                mean_arch = [sum(x) / len(x) for x in zip(*s["archetype_log"])]
                dom_idx = mean_arch.index(max(mean_arch))
                s["dominant_archetype"] = ARCHETYPE_NAMES[dom_idx]
                # Track archetype shifts (how many times dominant changes)
                step_doms = [ARCHETYPE_NAMES[w.index(max(w))]
                             for w in s["archetype_log"]]
                shifts = sum(1 for i in range(1, len(step_doms))
                             if step_doms[i] != step_doms[i-1])
                s["archetype_shifts"] = shifts
            else:
                s["dominant_archetype"] = None
                s["archetype_shifts"] = None

            model_results.append(s)
            arch_str = f" | arch={s['dominant_archetype']}" if s["dominant_archetype"] else ""
            print(f"      [{p['category']}] TTR={s['diversity']['ttr']:.3f}{arch_str}")

        results[key] = model_results

    return results


# ---------------------------------------------------------------------------
# 9. Metric M visualisation (Qwen suggestion #2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualise_metric_M(models_dict, tokenizer, device, max_seq_len=512):
    """
    For models with archetypal projection, extract the metric M for
    several prompts and:
      1. Compute PCA of M (flattened) to show prompt-dependent deformation
      2. Show which word-pairs become closer/farther in curved space
      3. Save visualisation to benchmark_results/
    """
    # Archetypal word groups -- words that should cluster under each archetype
    ARCHETYPE_WORDS = {
        "hero": ["hero", "sword", "quest", "battle", "victory", "courage", "journey"],
        "shadow": ["darkness", "fear", "enemy", "hidden", "secret", "evil", "night"],
        "anima": ["she", "beauty", "love", "grace", "maiden", "soul", "beloved"],
        "great_mother": ["river", "sea", "earth", "mother", "birth", "water", "womb"],
        "trickster": ["fool", "trick", "laughter", "mask", "cunning", "jest", "riddle"],
        "self": ["circle", "unity", "whole", "mandala", "center", "totality", "one"],
    }

    results = {}

    for key, (model, cfg) in models_dict.items():
        if not hasattr(model, "archetypal_proj"):
            continue

        name = cfg.get("_name", key)
        print(f"    {name}: extracting metric M...")

        # Get M for several different prompts
        test_prompts = [p[0] for p in THEMATIC_PROMPTS[:6]]
        M_list = []
        w_list = []

        for prompt in test_prompts:
            ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            out = _get_full_output(model, input_ids)
            if "M" in out:
                M_list.append(out["M"][0].cpu())  # [d, d]
                w_list.append(out["archetype_weights"][0].cpu().tolist())

        if not M_list:
            continue

        # Compute distances between archetype word groups in curved space
        word_distances = {}
        M_avg = torch.stack(M_list).mean(dim=0)  # average metric

        for arch_name, words in ARCHETYPE_WORDS.items():
            # Encode words and get embeddings
            word_ids = [tokenizer.encode(w) for w in words]
            valid_ids = [ids[0] for ids in word_ids if ids]  # first token

            if len(valid_ids) < 2:
                continue

            # Get token embeddings
            emb_weight = model.token_emb.weight.data.cpu()
            vecs = emb_weight[valid_ids]  # [n, d]

            # Euclidean distances
            eucl_dists = torch.cdist(vecs, vecs)
            eucl_avg = eucl_dists[eucl_dists > 0].mean().item()

            # Curved distances: x @ M gives position in curved space
            curved_vecs = vecs @ M_avg
            curved_dists = torch.cdist(curved_vecs, curved_vecs)
            curved_avg = curved_dists[curved_dists > 0].mean().item()

            word_distances[arch_name] = {
                "euclidean_avg": round(eucl_avg, 4),
                "curved_avg": round(curved_avg, 4),
                "ratio": round(curved_avg / max(eucl_avg, 1e-8), 4),
                "words": words[:5],
            }

        # PCA of M matrices (show how M changes per prompt)
        M_flat = torch.stack(M_list).reshape(len(M_list), -1)  # [n_prompts, d*d]
        M_centered = M_flat - M_flat.mean(dim=0)

        pca_info = None
        if M_flat.size(0) >= 2:
            U, S_vals, Vh = torch.linalg.svd(M_centered, full_matrices=False)
            coords_2d = (U[:, :2] * S_vals[:2]).tolist()
            variance_explained = (S_vals[:2] ** 2 / (S_vals ** 2).sum()).tolist()
            pca_info = {
                "coords": coords_2d,
                "prompts": test_prompts,
                "variance_explained": [round(v, 4) for v in variance_explained],
            }

        results[key] = {
            "word_distances": word_distances,
            "pca": pca_info,
            "archetype_weights_per_prompt": {
                p: w for p, w in zip(test_prompts, w_list)
            },
        }

        # Print summary
        print(f"      Word-group distances (curved/euclidean ratio):")
        for arch, d in word_distances.items():
            direction = "closer" if d["ratio"] < 1.0 else "farther"
            print(f"        {arch}: {d['ratio']:.3f} ({direction})")

        # Try to generate plot
        _try_plot_metric(results[key], name)

    return results


def _try_plot_metric(model_result, model_name):
    """Generate metric M visualisation if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("      (matplotlib not available, skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: word-group distance ratios ---
    wd = model_result["word_distances"]
    if wd:
        names = list(wd.keys())
        ratios = [wd[n]["ratio"] for n in names]
        colors = ["#2ecc71" if r < 1 else "#e74c3c" for r in ratios]
        ax = axes[0]
        bars = ax.barh(names, ratios, color=colors)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Curved / Euclidean distance ratio")
        ax.set_title(f"{model_name}: Archetypal curvature effect")
        for bar, r in zip(bars, ratios):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{r:.3f}", va="center", fontsize=9)

    # --- Right: PCA of M matrices ---
    pca = model_result.get("pca")
    if pca and pca["coords"]:
        coords = np.array(pca["coords"])
        prompts = pca["prompts"]
        ax = axes[1]
        ax.scatter(coords[:, 0], coords[:, 1], s=100, c=range(len(coords)),
                   cmap="viridis", zorder=5)
        for i, p in enumerate(prompts):
            ax.annotate(p[:20], (coords[i, 0], coords[i, 1]),
                        fontsize=8, ha="center", va="bottom")
        ve = pca["variance_explained"]
        ax.set_xlabel(f"PC1 ({ve[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ve[1]*100:.1f}%)")
        ax.set_title(f"{model_name}: Metric M across prompts (PCA)")

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = RESULTS_DIR / f"metric_M_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"      Saved: {out_path}")


# ---------------------------------------------------------------------------
# 10. Micro-narrative case study (Qwen suggestion #3)
# ---------------------------------------------------------------------------

def micro_narrative_study(models_dict, tokenizer, device, max_seq_len=512):
    """
    Generate completions for ambiguous prompts and analyse:
      - Symbolic resonance: does the model invoke symbols from the mythic corpus?
      - Archetypal coherence: do archetype weights track the narrative content?
      - Paradox preservation: does the model sustain tension or flatten it?
    """
    print("\n  [Micro-Narrative Case Study]")
    results = {}

    # Words from the mythic corpus that signal symbolic depth
    SYMBOLIC_WORDS = {
        "water": ["river", "sea", "water", "flood", "wave", "deep", "ocean",
                   "tears", "rain", "stream"],
        "fire": ["fire", "flame", "burning", "light", "sun", "blaze", "spark"],
        "death_rebirth": ["death", "died", "dead", "born", "birth", "rose",
                          "returned", "awoke", "descended", "grave"],
        "threshold": ["door", "gate", "threshold", "bridge", "passage",
                      "crossed", "entered", "opened"],
        "divine": ["god", "gods", "heaven", "holy", "sacred", "divine",
                    "prayer", "altar", "temple"],
    }

    for key, (model, cfg) in models_dict.items():
        name = cfg.get("_name", key)
        print(f"    {name}:")
        model_results = []

        for p in MICRO_NARRATIVE_PROMPTS:
            s = generate(model, tokenizer, p["prompt"], device,
                         max_tokens=GEN_LONG, max_seq_len=max_seq_len)
            s["description"] = p["description"]
            s["diversity"] = compute_diversity(s["generated_text"])

            # Symbolic resonance score
            text_lower = s["generated_text"].lower()
            words_in_text = set(re.findall(r"[a-z']+", text_lower))
            symbol_hits = {}
            total_hits = 0
            for category, symbol_words in SYMBOLIC_WORDS.items():
                hits = [w for w in symbol_words if w in words_in_text]
                symbol_hits[category] = hits
                total_hits += len(hits)
            s["symbolic_resonance"] = {
                "total_symbols": total_hits,
                "by_category": {k: len(v) for k, v in symbol_hits.items()},
                "symbol_words_found": {k: v for k, v in symbol_hits.items() if v},
            }

            # Archetype trajectory analysis
            if s["archetype_log"]:
                mean_arch = [sum(x) / len(x) for x in zip(*s["archetype_log"])]
                dom_idx = mean_arch.index(max(mean_arch))
                s["dominant_archetype"] = ARCHETYPE_NAMES[dom_idx]

                # Entropy of archetype distribution (high = diverse activations)
                w_tensor = torch.tensor(mean_arch)
                w_norm = w_tensor / w_tensor.sum()
                arch_entropy = -(w_norm * torch.log(w_norm + 1e-10)).sum().item()
                s["archetype_entropy"] = round(arch_entropy, 4)

                # Archetype shifts
                step_doms = [ARCHETYPE_NAMES[w.index(max(w))]
                             for w in s["archetype_log"]]
                s["archetype_shifts"] = sum(1 for i in range(1, len(step_doms))
                                            if step_doms[i] != step_doms[i-1])
            else:
                s["dominant_archetype"] = None
                s["archetype_entropy"] = None
                s["archetype_shifts"] = None

            model_results.append(s)

            sym_count = s["symbolic_resonance"]["total_symbols"]
            arch_str = ""
            if s["dominant_archetype"]:
                arch_str = f" | arch={s['dominant_archetype']}"
                arch_str += f" ent={s['archetype_entropy']:.2f}"
            print(f"      '{p['prompt'][:30]}...' "
                  f"symbols={sym_count} TTR={s['diversity']['ttr']:.3f}{arch_str}")

        results[key] = model_results

    return results


# ---------------------------------------------------------------------------
# Full model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name, model, cfg, tokenizer, dataloader, corpus_tokens, device):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    result = {"name": name, "config": cfg, "n_params": n_params}

    # 1. Perplexity
    print("  [1/7] Perplexity...")
    result["perplexity"] = compute_perplexity(model, dataloader, device)
    print(f"        PPL={result['perplexity']['perplexity']:.2f}, "
          f"CE={result['perplexity']['avg_loss']:.4f}")

    # 2. Long-context
    print("  [2/7] Long-context evaluation...")
    max_seq = cfg.get("seq_len", 512)
    result["long_context"] = long_context_evaluation(
        model, corpus_tokens, device, max_seq_len=max_seq
    )
    for ctx_sz, lc in result["long_context"].items():
        print(f"        ctx={ctx_sz}: PPL={lc['perplexity']:.2f}, "
              f"CE={lc['avg_loss']:.4f} ({lc['n_windows']} windows)")

    # 3. Generation at three lengths
    print("  [3/7] Generating literary samples...")
    result["generations"] = {}
    for gen_name, gen_len in [("short", GEN_SHORT), ("medium", GEN_MEDIUM), ("long", GEN_LONG)]:
        print(f"        {gen_name} ({gen_len} tokens):")
        samples = []
        for prompt, expected_arch, desc in THEMATIC_PROMPTS:
            s = generate(model, tokenizer, prompt, device,
                         max_tokens=gen_len, max_seq_len=max_seq)
            s["expected_archetype"] = expected_arch
            s["description"] = desc
            s["diversity"] = compute_diversity(s["generated_text"])
            samples.append(s)

            arch_info = ""
            if s["archetype_log"]:
                mean_arch = [sum(x) / len(x) for x in zip(*s["archetype_log"])]
                dom_idx = mean_arch.index(max(mean_arch))
                arch_info = f" | arch={ARCHETYPE_NAMES[dom_idx]}"
            print(f"          '{prompt}'{arch_info}")

        result["generations"][gen_name] = samples

    # 4. Aggregate diversity
    print("  [4/7] Diversity metrics (long generations)...")
    long_texts = " ".join(s["generated_text"] for s in result["generations"]["long"])
    result["diversity_long"] = compute_diversity(long_texts)
    print(f"        TTR={result['diversity_long']['ttr']:.4f}, "
          f"rep={result['diversity_long']['repetition_rate']:.4f}")

    # 5. Archetype analysis
    print("  [5/7] Archetype analysis...")
    result["archetypes"] = archetype_analysis(model, dataloader, device)
    if result["archetypes"]:
        top3 = sorted(result["archetypes"]["mean_weights"].items(),
                       key=lambda x: -x[1])[:3]
        print(f"        Top 3: {', '.join(f'{n}={v:.4f}' for n, v in top3)}")
    else:
        print("        (not applicable)")

    # 6. Entropy
    print("  [6/7] Token entropy...")
    result["mean_entropy"] = compute_token_entropy(model, dataloader, device)
    print(f"        H={result['mean_entropy']:.4f}")

    # 7. Speed
    print("  [7/7] Inference speed...")
    result["speed"] = measure_speed(model, device, seq_len=max_seq,
                                     vocab_size=cfg["vocab_size"])
    print(f"        {result['speed']['tokens_per_sec']:.0f} tokens/s "
          f"({result['speed']['avg_latency_ms']:.1f} ms/batch)")

    return result


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _escape_latex(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def generate_comparison_table(results):
    models = [(k, results[k]) for k in ["v1", "v2", "baseline"] if k in results]
    if len(models) < 2:
        return

    col_names = " & ".join(r"\textbf{%s}" % m["name"] for _, m in models)
    n_cols = len(models)

    latex = r"""\begin{table}[h]
\centering
\caption{Quantitative comparison on the mythic--Joycean corpus (""" + str(len(models)) + r""" models).
All models trained for 10 epochs with identical optimisation schedule.}
\label{tab:benchmark}
\begin{tabular}{l""" + "c" * n_cols + r"""}
\toprule
\textbf{Metric} & """ + col_names + r""" \\
\midrule
"""

    row = " & ".join(f"{m['n_params']:,}" for _, m in models)
    latex += f"Parameters & {row} \\\\\n"

    for field, label in [("d_model", r"$d_{{\text{{model}}}}$"), ("n_layers", "Layers"),
                         ("n_heads", "Heads"), ("d_ff", r"$d_{{\text{{ff}}}}$")]:
        row = " & ".join(str(m["config"].get(field, "---")) for _, m in models)
        latex += f"{label} & {row} \\\\\n"

    latex += r"\midrule" + "\n"

    row = " & ".join(f"{m['perplexity']['perplexity']:.2f}" for _, m in models)
    latex += f"Perplexity & {row} \\\\\n"
    row = " & ".join(f"{m['perplexity']['avg_loss']:.4f}" for _, m in models)
    latex += f"CE Loss & {row} \\\\\n"

    latex += r"\midrule" + "\n"
    for ctx_sz in CONTEXT_SIZES:
        row_parts = []
        for _, m in models:
            lc = m.get("long_context", {}).get(ctx_sz, {})
            row_parts.append(f"{lc.get('perplexity', '---')}")
        latex += f"PPL (ctx={ctx_sz}) & {' & '.join(row_parts)} \\\\\n"

    latex += r"\midrule" + "\n"
    row = " & ".join(f"{m.get('diversity_long', {}).get('ttr', 0):.4f}" for _, m in models)
    latex += f"Type-Token Ratio & {row} \\\\\n"
    row = " & ".join(f"{m.get('diversity_long', {}).get('repetition_rate', 0):.4f}" for _, m in models)
    latex += f"Repetition Rate & {row} \\\\\n"

    row = " & ".join(f"{m.get('mean_entropy', 0):.4f}" for _, m in models)
    latex += f"Mean Entropy & {row} \\\\\n"

    latex += r"\midrule" + "\n"
    row = " & ".join(f"{m.get('speed', {}).get('tokens_per_sec', 0):.0f}" for _, m in models)
    latex += f"Tokens/s & {row} \\\\\n"

    latex += r"\midrule" + "\n"
    feat = " & ".join("---" if k == "baseline" else r"$\checkmark$" for k, _ in models)
    latex += f"Archetype Tensors & {feat} \\\\\n"
    mem = " & ".join(r"$\checkmark$" if k == "v2" else "---" for k, _ in models)
    latex += f"Neural Memory & {mem} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    out = RESULTS_DIR / "comparison_table.tex"
    out.write_text(latex)
    print(f"  LaTeX comparison table: {out}")


def generate_long_context_table(results):
    models = [(k, results[k]) for k in ["v1", "v2", "baseline"]
              if k in results and isinstance(results[k], dict)]
    if len(models) < 2:
        return

    col_names = " & ".join(r"\textbf{%s}" % m["name"] for _, m in models)
    n_cols = len(models)

    latex = r"""\begin{table}[h]
\centering
\caption{Long-context utilisation: perplexity on next-""" + str(PREDICT_WINDOW) + r""" tokens
after feeding contexts of increasing length. Lower is better.}
\label{tab:longctx}
\begin{tabular}{l""" + "c" * n_cols + r"""}
\toprule
\textbf{Context Length} & """ + col_names + r""" \\
\midrule
"""
    for ctx_sz in CONTEXT_SIZES:
        row_parts = []
        for _, m in models:
            lc = m.get("long_context", {}).get(ctx_sz, {})
            ppl = lc.get("perplexity", "---")
            ce = lc.get("avg_loss", "---")
            row_parts.append(f"{ppl} ({ce})")
        latex += f"{ctx_sz} tokens & {' & '.join(row_parts)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    out = RESULTS_DIR / "long_context_table.tex"
    out.write_text(latex)
    print(f"  LaTeX long-context table: {out}")


def generate_samples_table(results):
    models = [(k, results[k]) for k in ["v1", "v2", "baseline"]
              if k in results and isinstance(results[k], dict)]
    if len(models) < 2:
        return

    col_names = " & ".join(r"\textbf{%s}" % m["name"] for _, m in models)
    n_cols = len(models)
    col_width = 13.0 / n_cols

    latex = r"""\begin{table*}[t]
\centering
\caption{Generation samples from thematic prompts (temperature 0.8, nucleus $p=0.9$, """ + str(GEN_MEDIUM) + r""" tokens).
Bold text is the prompt. Text truncated at 250 characters.}
\label{tab:samples}
\small
\begin{tabular}{p{2cm}""" + (r"p{%.1fcm}" % col_width) * n_cols + r"""}
\toprule
\textbf{Prompt} & """ + col_names + r""" \\
\midrule
"""
    for i in range(min(5, len(THEMATIC_PROMPTS))):
        prompt_tex = _escape_latex(THEMATIC_PROMPTS[i][0])
        texts = []
        for _, m in models:
            gens = m.get("generations", {}).get("medium", [])
            if i < len(gens):
                texts.append(_escape_latex(gens[i]["generated_text"][:250]))
            else:
                texts.append("---")
        cols = " & ".join(texts)
        latex += r"\textbf{" + prompt_tex + "} & " + cols + r" \\" + "\n"
        latex += r"\midrule" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    out = RESULTS_DIR / "samples_table.tex"
    out.write_text(latex)
    print(f"  LaTeX samples table: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    v1_ckpt="checkpoints/final.pt",
    v2_ckpt="checkpoints_v2/final.pt",
    baseline_ckpt="checkpoints_baseline/final.pt",
):
    device = DEVICE
    print(f"Device: {device}")
    print(f"Generation lengths: short={GEN_SHORT}, medium={GEN_MEDIUM}, long={GEN_LONG}")
    print(f"Long-context sizes: {CONTEXT_SIZES}, predict window: {PREDICT_WINDOW}")

    # --- Load corpus and tokenizer ---
    corpus = load_cached_corpus()
    vocab_path = DATA_DIR / "vocab.txt"
    tokenizer = WordTokenizer()
    if vocab_path.exists():
        tokenizer.load(vocab_path)
    else:
        tokenizer.build(corpus)
    print(f"Vocabulary: {tokenizer.vocab_size:,} tokens")

    dataloader = build_dataloader(
        corpus, tokenizer, seq_len=512, batch_size=4, num_workers=0,
    )

    corpus_tokens = []
    for _, text in corpus:
        corpus_tokens.extend(tokenizer.encode(text))
        corpus_tokens.append(tokenizer.eot_id)
    print(f"Corpus tokens: {len(corpus_tokens):,}")

    results = {
        "device": device,
        "vocab_size": tokenizer.vocab_size,
        "gen_lengths": {"short": GEN_SHORT, "medium": GEN_MEDIUM, "long": GEN_LONG},
        "context_sizes": CONTEXT_SIZES,
        "predict_window": PREDICT_WINDOW,
    }

    # --- Evaluate each model (quantitative) ---
    model_specs = [
        ("v1", "DisattentionFormer V1", v1_ckpt, load_v1_model),
        ("v2", "DisattentionFormer V2 (Titans)", v2_ckpt, load_v2_model),
        ("baseline", "Baseline Transformer", baseline_ckpt, load_baseline_model),
    ]

    loaded_models = {}  # keep models for qualitative analysis

    for key, name, ckpt_path, loader_fn in model_specs:
        if not os.path.exists(ckpt_path):
            print(f"\n  [{name}] Checkpoint not found: {ckpt_path} -- skipping")
            continue

        model, cfg = loader_fn(ckpt_path, device)
        cfg["_name"] = name
        result = evaluate_model(name, model, cfg, tokenizer, dataloader,
                                corpus_tokens, device)
        results[key] = result
        loaded_models[key] = (model, cfg)

    # --- Qualitative evaluations (Qwen suggestions) ---
    if loaded_models:
        print(f"\n{'=' * 60}")
        print("  QUALITATIVE EVALUATIONS")
        print(f"{'=' * 60}")

        # 8. Narrative error analysis
        results["narrative_errors"] = narrative_error_analysis(
            loaded_models, tokenizer, device
        )

        # 9. Metric M visualisation
        print("\n  [Metric M Visualisation]")
        results["metric_m"] = visualise_metric_M(
            loaded_models, tokenizer, device
        )

        # 10. Micro-narrative case study
        results["micro_narratives"] = micro_narrative_study(
            loaded_models, tokenizer, device
        )

    # --- Free models ---
    del loaded_models
    if device == "mps":
        torch.mps.empty_cache()

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    available = [(k, results[k]) for k in ["v1", "v2", "baseline"] if k in results]
    if available:
        header = f"{'Metric':<30}" + "".join(f"{m['name']:>25}" for _, m in available)
        print(header)
        print("-" * len(header))

        row = "".join(f"{m['n_params']:>25,}" for _, m in available)
        print(f"{'Parameters':<30}{row}")

        row = "".join(f"{m['perplexity']['perplexity']:>25.2f}" for _, m in available)
        print(f"{'Perplexity':<30}{row}")

        row = "".join(f"{m['perplexity']['avg_loss']:>25.4f}" for _, m in available)
        print(f"{'CE Loss':<30}{row}")

        for ctx_sz in CONTEXT_SIZES:
            row = "".join(
                f"{m.get('long_context', {}).get(ctx_sz, {}).get('perplexity', 0):>25.2f}"
                for _, m in available
            )
            print(f"{'PPL (ctx=' + str(ctx_sz) + ')':<30}{row}")

        row = "".join(f"{m.get('diversity_long', {}).get('ttr', 0):>25.4f}"
                      for _, m in available)
        print(f"{'TTR (long gen)':<30}{row}")

        row = "".join(f"{m.get('diversity_long', {}).get('repetition_rate', 0):>25.4f}"
                      for _, m in available)
        print(f"{'Repetition (long gen)':<30}{row}")

        row = "".join(f"{m.get('mean_entropy', 0):>25.4f}" for _, m in available)
        print(f"{'Mean Entropy':<30}{row}")

        row = "".join(f"{m.get('speed', {}).get('tokens_per_sec', 0):>25.0f}"
                      for _, m in available)
        print(f"{'Tokens/s':<30}{row}")

    # --- Save ---
    out_path = RESULTS_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults JSON: {out_path}")

    # --- LaTeX tables ---
    print("\nGenerating LaTeX tables...")
    generate_comparison_table(results)
    generate_long_context_table(results)
    generate_samples_table(results)

    print("\nBenchmark complete.")
    return results


if __name__ == "__main__":
    run_benchmark()
