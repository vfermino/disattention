"""
Download, annotate, and prepare James Joyce's works for linguistics-aware training.

Pipeline:
1. Download texts from Project Gutenberg
2. Strip Gutenberg headers/footers
3. Run spaCy NLP pipeline to extract linguistic features
4. For each character position, assign the linguistic features of its parent word
5. Build vocabularies for all layers (char, POS, dep, morph, shape)
6. Save tokenized + annotated train/validation splits
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import requests
import spacy
from tqdm import tqdm

from config import DataConfig
from tokenizer import LinguisticTokenizer


def download_text(url: str, dest: Path, max_retries: int = 5) -> str:
    """Download a text file from a URL with retries."""
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        with open(dest, "r", encoding="utf-8") as f:
            return f.read()

    import time as _time

    for attempt in range(max_retries):
        try:
            print(f"  Downloading: {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            text = response.text

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(text)

            return text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Timeout/connection error, retrying in {wait}s... ({e})")
                _time.sleep(wait)
            else:
                raise


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    start_patterns = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"Produced by",
    ]
    start_idx = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_idx = match.end()
            next_nl = text.find("\n", start_idx)
            if next_nl != -1:
                start_idx = next_nl + 1
            break

    end_patterns = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
        r"End of (?:the )?Project Gutenberg",
    ]
    end_idx = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_idx = match.start()
            break

    return text[start_idx:end_idx].strip()


def annotate_text(nlp, text: str, chunk_size: int = 100000) -> dict:
    """
    Run spaCy on text and produce character-aligned linguistic annotations.

    For each character, we assign the linguistic features of the spaCy token
    that contains it. Whitespace/non-token characters get a special "_WS" tag.

    Returns dict with parallel lists:
        chars: list of single characters
        pos: list of POS tags (one per character)
        dep: list of dependency relations
        morph: list of primary morphological feature strings
        shape: list of word shape strings
    """
    result = {"chars": [], "pos": [], "dep": [], "morph": [], "shape": []}

    # Process in chunks to manage memory (Ulysses is large)
    for start in tqdm(
        range(0, len(text), chunk_size), desc="  Annotating", leave=False
    ):
        chunk = text[start : start + chunk_size]
        doc = nlp(chunk)

        # Build character-to-token mapping
        char_annotations = [("_WS", "_WS", "_WS", "_WS")] * len(chunk)

        for token in doc:
            # Get primary morphological feature (most informative single feature)
            morph_str = str(token.morph) if str(token.morph) else "_NONE"
            # Simplify: take first morph feature or use _NONE
            morph_parts = morph_str.split("|")
            primary_morph = morph_parts[0] if morph_parts[0] else "_NONE"

            shape = token.shape_ if token.shape_ else "_NONE"

            for i in range(token.idx, token.idx + len(token.text)):
                if i < len(chunk):
                    char_annotations[i] = (token.pos_, token.dep_, primary_morph, shape)

        for i, ch in enumerate(chunk):
            pos, dep, morph, shape = char_annotations[i]
            result["chars"].append(ch)
            result["pos"].append(pos)
            result["dep"].append(dep)
            result["morph"].append(morph)
            result["shape"].append(shape)

    return result


def prepare_data(config: DataConfig | None = None) -> None:
    """Download, annotate, and prepare the full Joyce corpus."""
    if config is None:
        config = DataConfig()

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    # Load spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    # Increase max length for Ulysses
    nlp.max_length = 2_000_000

    corpus_parts = []

    print("Downloading James Joyce's works...")
    for name, url in tqdm(config.sources.items(), desc="Works"):
        dest = config.raw_dir / f"{name}.txt"
        raw_text = download_text(url, dest)
        cleaned = strip_gutenberg(raw_text)

        if len(cleaned) < 100:
            print(
                f"  Warning: {name} too short after cleaning ({len(cleaned)} chars), skipping."
            )
            continue

        print(f"  {name}: {len(cleaned):,} characters")
        corpus_parts.append(cleaned)

    full_corpus = "\n\n\n".join(corpus_parts)
    print(f"\nTotal corpus: {len(full_corpus):,} characters")

    # Save raw corpus
    corpus_path = config.processed_dir / config.corpus_file
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    # Annotate with linguistic features
    print("\nRunning linguistic annotation (this may take a few minutes)...")
    annotations = annotate_text(nlp, full_corpus)
    print(f"Annotated {len(annotations['chars']):,} character positions")

    # Print some statistics about feature distributions
    from collections import Counter

    for layer in ("pos", "dep"):
        counts = Counter(annotations[layer])
        top10 = counts.most_common(10)
        print(f"\n  Top {layer.upper()} tags: {top10}")

    # Build tokenizer with all vocabularies
    tokenizer = LinguisticTokenizer()
    tokenizer.build_vocab("char", annotations["chars"])
    tokenizer.build_vocab("pos", annotations["pos"])
    tokenizer.build_vocab("dep", annotations["dep"])
    tokenizer.build_vocab("morph", annotations["morph"])
    tokenizer.build_vocab("shape", annotations["shape"])

    print(f"\nVocabulary sizes: {tokenizer}")

    vocab_path = config.processed_dir / config.vocab_file
    tokenizer.save(vocab_path)
    print(f"Saved vocabularies to {vocab_path}")

    # Encode all layers
    encoded = {
        "char": tokenizer.encode_layer("char", annotations["chars"]),
        "pos": tokenizer.encode_layer("pos", annotations["pos"]),
        "dep": tokenizer.encode_layer("dep", annotations["dep"]),
        "morph": tokenizer.encode_layer("morph", annotations["morph"]),
        "shape": tokenizer.encode_layer("shape", annotations["shape"]),
    }

    # Split
    split_idx = int(len(encoded["char"]) * config.train_split)
    train_data = {k: v[:split_idx] for k, v in encoded.items()}
    val_data = {k: v[split_idx:] for k, v in encoded.items()}

    import torch

    torch.save(train_data, config.processed_dir / "train.pt")
    torch.save(val_data, config.processed_dir / "val.pt")
    print(f"\nTrain: {len(train_data['char']):,} tokens")
    print(f"Val:   {len(val_data['char']):,} tokens")
    print("\nData preparation complete.")


if __name__ == "__main__":
    prepare_data()
