"""
Text generation script for the linguistics-aware JoyceGPT.

During generation, we use spaCy to annotate the prompt and generated text
so far, providing linguistic features as input conditioning at each step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import spacy
import torch
import torch.nn.functional as F

from config import Config, get_device
from model import JoyceGPT
from tokenizer import LinguisticTokenizer


def annotate_for_generation(nlp, text: str, tokenizer: LinguisticTokenizer) -> dict:
    """
    Run spaCy on text and return encoded linguistic feature sequences.
    Used to provide conditioning context during generation.
    """
    doc = nlp(text)

    chars = list(text)
    pos_list = ["_WS"] * len(text)
    dep_list = ["_WS"] * len(text)
    morph_list = ["_WS"] * len(text)
    shape_list = ["_WS"] * len(text)

    for token in doc:
        morph_str = str(token.morph) if str(token.morph) else "_NONE"
        morph_parts = morph_str.split("|")
        primary_morph = morph_parts[0] if morph_parts[0] else "_NONE"
        shape = token.shape_ if token.shape_ else "_NONE"

        for i in range(token.idx, token.idx + len(token.text)):
            if i < len(text):
                pos_list[i] = token.pos_
                dep_list[i] = token.dep_
                morph_list[i] = primary_morph
                shape_list[i] = shape

    return {
        "char": tokenizer.encode_layer("char", chars),
        "pos": tokenizer.encode_layer("pos", pos_list),
        "dep": tokenizer.encode_layer("dep", dep_list),
        "morph": tokenizer.encode_layer("morph", morph_list),
        "shape": tokenizer.encode_layer("shape", shape_list),
    }


@torch.no_grad()
def generate(
    model: JoyceGPT,
    tokenizer: LinguisticTokenizer,
    nlp,
    prompt: str,
    max_new_tokens: int = 1000,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
    reannotate_interval: int = 50,
) -> str:
    """
    Generate text with linguistic conditioning.

    Every `reannotate_interval` characters, we re-run spaCy on the generated
    text so far to update the linguistic features. This is a tradeoff between
    generation speed and annotation accuracy.
    """
    model.eval()

    generated_text = prompt
    steps_since_annotation = 0

    # Initial annotation
    encoded = annotate_for_generation(nlp, generated_text, tokenizer)

    for step in range(max_new_tokens):
        # Re-annotate periodically for fresh linguistic context
        if steps_since_annotation >= reannotate_interval:
            encoded = annotate_for_generation(nlp, generated_text, tokenizer)
            steps_since_annotation = 0

        # Prepare input tensors (crop to max_seq_len)
        max_len = model.max_seq_len
        char_ids = torch.tensor(
            [encoded["char"][-max_len:]], dtype=torch.long, device=device
        )
        pos_ids = torch.tensor(
            [encoded["pos"][-max_len:]], dtype=torch.long, device=device
        )
        dep_ids = torch.tensor(
            [encoded["dep"][-max_len:]], dtype=torch.long, device=device
        )
        morph_ids = torch.tensor(
            [encoded["morph"][-max_len:]], dtype=torch.long, device=device
        )
        shape_ids = torch.tensor(
            [encoded["shape"][-max_len:]], dtype=torch.long, device=device
        )

        # Forward pass
        result = model(
            idx=char_ids,
            pos_tags=pos_ids,
            dep_tags=dep_ids,
            morph_tags=morph_ids,
            shape_tags=shape_ids,
        )
        logits = result["char_logits"][:, -1, :]

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            kth_vals = torch.topk(logits, top_k_val, dim=-1).values[:, -1:]
            logits = torch.where(
                logits < kth_vals, torch.full_like(logits, float("-inf")), logits
            )

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_char_idx = next_token.item()

        # Decode the character
        next_char = tokenizer.decode_chars([next_char_idx])
        generated_text += next_char

        # Append to encoded sequences (use _WS as placeholder until re-annotation)
        encoded["char"].append(next_char_idx)
        unk = tokenizer.vocabs["pos"].get("_WS", 1)
        encoded["pos"].append(tokenizer.vocabs["pos"].get("_WS", unk))
        encoded["dep"].append(tokenizer.vocabs["dep"].get("_WS", unk))
        encoded["morph"].append(tokenizer.vocabs["morph"].get("_WS", unk))
        encoded["shape"].append(tokenizer.vocabs["shape"].get("_WS", unk))

        steps_since_annotation += 1

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with JoyceGPT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Stately, plump Buck Mulligan",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1000, help="Maximum characters to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--reannotate-interval",
        type=int,
        default=50,
        help="Re-run spaCy every N generated chars",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load tokenizer
    config = Config()
    vocab_path = config.data.processed_dir / config.data.vocab_file
    if not vocab_path.exists():
        print("Error: Vocabulary not found. Run prepare_data.py first.")
        sys.exit(1)

    tokenizer = LinguisticTokenizer()
    tokenizer.load(vocab_path)

    # Load spaCy for generation-time annotation
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = checkpoint["config"]

    model = JoyceGPT(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
        bias=model_cfg["bias"],
        n_pos_tags=model_cfg["n_pos_tags"],
        n_dep_tags=model_cfg["n_dep_tags"],
        n_morph_tags=model_cfg["n_morph_tags"],
        n_shape_tags=model_cfg["n_shape_tags"],
        d_ling_emb=model_cfg["d_ling_emb"],
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded ({model.count_parameters():,} parameters)")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        nlp=nlp,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        reannotate_interval=args.reannotate_interval,
    )

    print(output)
    print("-" * 60)
    print(f"Generated {len(output) - len(args.prompt)} new characters")


if __name__ == "__main__":
    main()
