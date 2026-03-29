"""
Generation script for the DisattentionFormer.

Nucleus sampling with archetype diagnostics.
Shows which archetypal field is active during generation,
so you can observe the Viconian cycle in real time.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from desatencao_data import WordTokenizer, DATA_DIR
from desatencao_model import DisattentionFormer


def get_device() -> torch.device:
    """Select the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def generate(
    model: DisattentionFormer,
    tokenizer: WordTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device = None,
) -> tuple:
    """
    Generate text with nucleus sampling and archetype diagnostics.

    Returns:
        (generated_text, metadata)

    metadata["archetype_log"] records the dominant archetype per generated
    token -- a diagnostic of the active archetypal field during generation.
    """
    if device is None:
        device = next(model.parameters()).device

    archetype_names = list(model.archetypal_proj.archetype_names)
    model.eval()

    ids = (
        torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    )
    prompt_len = ids.size(1)
    log = []

    max_pos = model.pos_emb.num_embeddings

    for _ in range(max_new_tokens):
        # Truncate to max sequence length the model can handle
        ids_ctx = ids[:, -max_pos:]
        out = model(ids_ctx, return_internals=False)

        logits = out["logits"][:, -1, :] / max(temperature, 1e-8)
        w = out["archetype_weights"][0]

        log.append(
            {
                "position": ids.size(1),
                "dominant": archetype_names[w.argmax().item()],
                "weights": w.cpu().tolist(),
            }
        )

        # -- Nucleus (top-p) sampling with numerical stability ---------------
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        # Stabilize probabilities before multinomial sampling
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp(min=0.0)
        if probs.sum() < 1e-8:  # fallback: uniform distribution
            probs = torch.ones_like(probs) / probs.size(-1)
        else:
            probs = probs / probs.sum()

        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.eot_id:
            break

    generated_text = tokenizer.decode(ids[0, prompt_len:].tolist())
    full_text = tokenizer.decode(ids[0].tolist())

    # Summarize archetype activity
    dominant_counts = {}
    for entry in log:
        d = entry["dominant"]
        dominant_counts[d] = dominant_counts.get(d, 0) + 1

    dominant_overall = max(dominant_counts, key=dominant_counts.get) if log else "none"

    metadata = {
        "archetype_log": log,
        "dominant_overall": dominant_overall,
        "dominant_counts": dominant_counts,
        "tokens_generated": len(log),
    }

    return full_text, generated_text, metadata


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> DisattentionFormer:
    """Load a trained DisattentionFormer from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]

    # Load archetype tensors
    tensor_path = "archetype_tensors.pt"
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(
            f"Archetype tensors not found at {tensor_path}. "
            "Run build_archetypes.py first."
        )
    archetype_tensors = torch.load(tensor_path, map_location=device, weights_only=True)
    archetype_tensors = {k: v.to(device) for k, v in archetype_tensors.items()}

    model = DisattentionFormer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["seq_len"],
        archetype_tensors=archetype_tensors,
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} parameters")
    print(
        f"  d_model={config['d_model']}, n_layers={config['n_layers']}, "
        f"n_heads={config['n_heads']}"
    )
    return model


def print_archetype_report(metadata: dict):
    """Print a diagnostic report of archetype activity during generation."""
    print("\n--- Archetype Diagnostics ---")
    print(f"Tokens generated: {metadata['tokens_generated']}")
    print(f"Dominant archetype (overall): {metadata['dominant_overall']}")
    print("\nArchetype frequency:")
    for name, count in sorted(metadata["dominant_counts"].items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"  {name:>16s}: {count:3d} {bar}")

    # Show archetype transitions (first 30 tokens)
    log = metadata["archetype_log"]
    n_show = min(30, len(log))
    if n_show > 0:
        print(f"\nArchetypal field (first {n_show} tokens):")
        # Single-char symbols for compact display of 16 archetypes
        symbols = {
            "self": "O",  # Ouroboros / totality
            "shadow": "S",
            "anima": "A",
            "animus": "U",  # mascUline logos
            "great_mother": "M",
            "great_father": "F",
            "hero": "H",
            "wise_old_man": "W",
            "persona": "P",
            "puer_aeternus": "Y",  # Youth
            "senex": "X",  # old/Saturn
            "trickster": "T",
            "kore": "K",
            "divine_child": "C",
            "spirit": "I",  # Inspiration / pneuma
            "quaternity": "Q",
        }
        seq = "".join(symbols.get(e["dominant"], "?") for e in log[:n_show])
        print(f"  {seq}")
        legend = " ".join(f"{v}={k}" for k, v in symbols.items())
        print(f"  Legend: {legend}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with the DisattentionFormer"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In the beginning",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Suppress archetype diagnostic output",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load word tokenizer
    vocab_path = DATA_DIR / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_path}. "
            "Run desatencao_data.py or desatencao_train.py first."
        )
    tokenizer = WordTokenizer().load(vocab_path)
    print(f"Vocabulary: {tokenizer.vocab_size:,} words")

    model = load_model(args.checkpoint, device)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Temperature: {args.temperature}, top_p: {args.top_p}")
    print(f"Max new tokens: {args.max_tokens}")
    print("\n--- Generated Text ---\n")

    full_text, generated_text, metadata = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print(full_text)

    if not args.no_diagnostics:
        print_archetype_report(metadata)


if __name__ == "__main__":
    main()
