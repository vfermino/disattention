"""
Gutenberg corpus preparation for 85M-parameter model training.

Produces tokenized binary files for three configurations:
  - Word-level (V1/V2): builds vocab from corpus, caps at MAX_VOCAB
  - BPE (Baseline): uses tiktoken GPT-2 encoding (50,257 tokens)

All outputs go to data/gutenberg/ with no overwrites of existing files.

Usage:
    python3 gutenberg_prepare.py                     # default: 2GB text limit
    python3 gutenberg_prepare.py --max-gb 5          # use up to 5GB
    python3 gutenberg_prepare.py --max-gb 0          # use ALL text (24GB)
"""

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_PATH = Path("data/gutenberg/corpus.txt")
OUT_DIR = Path("data/gutenberg")

MAX_VOCAB = 80_000          # word-level vocab cap (top-N by frequency)
MIN_FREQ = 5                # minimum frequency for word inclusion
TRAIN_RATIO = 0.95          # 95% train, 5% val
CHUNK_SIZE = 1024 * 1024    # read 1MB at a time for streaming

# Special tokens (must match desatencao_data.py)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOT_TOKEN = "<eot>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOT_TOKEN]


def tokenize_words(text: str) -> list[str]:
    """Split text into word tokens and punctuation (same as desatencao_data.py)."""
    return re.findall(r"[A-Za-z'\u2019-]+|[0-9]+|[^\s]", text)


# ---------------------------------------------------------------------------
# Step 1: Build word-level vocabulary by streaming through corpus
# ---------------------------------------------------------------------------

def build_word_vocab(corpus_path: Path, max_bytes: int) -> tuple[list[str], dict[str, int]]:
    """Stream through corpus, count word frequencies, build vocab."""
    print("Building word-level vocabulary...")
    counter = Counter()
    bytes_read = 0

    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            bytes_read += len(chunk.encode("utf-8", errors="replace"))
            words = tokenize_words(chunk)
            counter.update(words)

            if bytes_read % (100 * CHUNK_SIZE) == 0:
                print(f"  Scanned {bytes_read / 1e9:.1f}GB, {len(counter):,} unique words...",
                      flush=True)

            if max_bytes > 0 and bytes_read >= max_bytes:
                break

    print(f"  Total unique words (raw): {len(counter):,}")
    print(f"  Words with freq >= {MIN_FREQ}: {sum(1 for c in counter.values() if c >= MIN_FREQ):,}")

    # Filter by min frequency, take top MAX_VOCAB by count
    filtered = [(w, c) for w, c in counter.items() if c >= MIN_FREQ]
    filtered.sort(key=lambda x: -x[1])
    if len(filtered) > MAX_VOCAB:
        filtered = filtered[:MAX_VOCAB]

    # Build vocab: special tokens first, then words by frequency
    idx2word = list(SPECIAL_TOKENS) + [w for w, _ in filtered]
    word2idx = {w: i for i, w in enumerate(idx2word)}

    print(f"  Final vocabulary: {len(idx2word):,} tokens "
          f"(3 special + {len(filtered):,} words)")

    return idx2word, word2idx


# ---------------------------------------------------------------------------
# Step 2: Tokenize corpus into memory-mapped binary files
# ---------------------------------------------------------------------------

def tokenize_word_level(corpus_path: Path, word2idx: dict, max_bytes: int,
                        unk_id: int, eot_id: int) -> np.ndarray:
    """Tokenize entire corpus to numpy array of int32 token IDs."""
    print("Tokenizing (word-level)...")
    all_ids = []
    bytes_read = 0
    text_count = 0

    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        # The corpus has texts separated by \n\n========== (from download script)
        buffer = ""
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                # Process remaining buffer
                if buffer.strip():
                    words = tokenize_words(buffer)
                    ids = [word2idx.get(w, unk_id) for w in words]
                    all_ids.extend(ids)
                    all_ids.append(eot_id)
                    text_count += 1
                break

            bytes_read += len(chunk.encode("utf-8", errors="replace"))
            buffer += chunk

            # Split on text separator
            while "\n\n==========" in buffer:
                idx = buffer.index("\n\n==========")
                text = buffer[:idx]
                # Skip the separator line
                rest_start = buffer.index("\n", idx + 2)
                buffer = buffer[rest_start + 1:]

                if text.strip():
                    words = tokenize_words(text)
                    ids = [word2idx.get(w, unk_id) for w in words]
                    all_ids.extend(ids)
                    all_ids.append(eot_id)
                    text_count += 1

                    if text_count % 1000 == 0:
                        print(f"  {text_count:,} texts, {len(all_ids):,} tokens, "
                              f"{bytes_read / 1e9:.1f}GB...", flush=True)

            if max_bytes > 0 and bytes_read >= max_bytes:
                # Process remaining buffer
                if buffer.strip():
                    words = tokenize_words(buffer)
                    ids = [word2idx.get(w, unk_id) for w in words]
                    all_ids.extend(ids)
                    all_ids.append(eot_id)
                    text_count += 1
                break

    print(f"  Done: {text_count:,} texts, {len(all_ids):,} tokens")
    return np.array(all_ids, dtype=np.int32)


def tokenize_bpe(corpus_path: Path, max_bytes: int) -> np.ndarray:
    """Tokenize corpus using tiktoken GPT-2 BPE."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    print("Tokenizing (BPE / tiktoken GPT-2)...")
    all_ids = []
    bytes_read = 0
    text_count = 0
    eot_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        buffer = ""
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                if buffer.strip():
                    ids = enc.encode(buffer)
                    all_ids.extend(ids)
                    all_ids.append(eot_id)
                    text_count += 1
                break

            bytes_read += len(chunk.encode("utf-8", errors="replace"))
            buffer += chunk

            while "\n\n==========" in buffer:
                idx = buffer.index("\n\n==========")
                text = buffer[:idx]
                rest_start = buffer.index("\n", idx + 2)
                buffer = buffer[rest_start + 1:]

                if text.strip():
                    ids = enc.encode(text)
                    all_ids.extend(ids)
                    all_ids.append(eot_id)
                    text_count += 1

                    if text_count % 1000 == 0:
                        print(f"  {text_count:,} texts, {len(all_ids):,} tokens, "
                              f"{bytes_read / 1e9:.1f}GB...", flush=True)

            if max_bytes > 0 and bytes_read >= max_bytes:
                if buffer.strip():
                    ids = enc.encode(buffer)
                    all_ids.extend(ids)
                    all_ids.append(eot_id)
                    text_count += 1
                break

    print(f"  Done: {text_count:,} texts, {len(all_ids):,} tokens")
    return np.array(all_ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# Step 3: Split and save
# ---------------------------------------------------------------------------

def save_splits(tokens: np.ndarray, prefix: str, out_dir: Path):
    """Save train/val splits as .bin files (memory-mappable)."""
    n = len(tokens)
    split_idx = int(n * TRAIN_RATIO)

    train_path = out_dir / f"{prefix}_train.bin"
    val_path = out_dir / f"{prefix}_val.bin"

    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"  {prefix}_train.bin: {len(train_tokens):,} tokens ({train_path.stat().st_size / 1e6:.1f}MB)")
    print(f"  {prefix}_val.bin:   {len(val_tokens):,} tokens ({val_path.stat().st_size / 1e6:.1f}MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Gutenberg corpus for 85M training")
    parser.add_argument("--max-gb", type=float, default=2.0,
                        help="Max GB of text to use (0 = all). Default: 2.0")
    parser.add_argument("--skip-bpe", action="store_true",
                        help="Skip BPE tokenization (only build word-level)")
    parser.add_argument("--skip-word", action="store_true",
                        help="Skip word-level tokenization (only build BPE)")
    args = parser.parse_args()

    max_bytes = int(args.max_gb * 1e9) if args.max_gb > 0 else 0

    if not CORPUS_PATH.exists():
        print(f"Error: {CORPUS_PATH} not found. Run download_gutenberg.py first.")
        sys.exit(1)

    corpus_size = CORPUS_PATH.stat().st_size
    print(f"Corpus: {CORPUS_PATH} ({corpus_size / 1e9:.1f}GB)")
    if max_bytes > 0:
        print(f"Using first {args.max_gb}GB of text")
    else:
        print("Using ALL text")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Word-level tokenization (for V1/V2) ----
    if not args.skip_word:
        print(f"\n{'='*60}")
        print("  WORD-LEVEL TOKENIZATION (V1/V2)")
        print(f"{'='*60}")

        idx2word, word2idx = build_word_vocab(CORPUS_PATH, max_bytes)

        # Save vocab
        vocab_path = OUT_DIR / "word_vocab.txt"
        vocab_path.write_text("\n".join(idx2word), encoding="utf-8")
        print(f"  Saved vocab: {vocab_path}")

        # Also save as JSON for easy inspection
        vocab_json_path = OUT_DIR / "word_vocab.json"
        with open(vocab_json_path, "w") as f:
            json.dump({"vocab_size": len(idx2word), "idx2word": idx2word}, f)
        print(f"  Saved vocab JSON: {vocab_json_path}")

        unk_id = word2idx[UNK_TOKEN]
        eot_id = word2idx[EOT_TOKEN]
        tokens = tokenize_word_level(CORPUS_PATH, word2idx, max_bytes, unk_id, eot_id)

        print(f"\n  Saving word-level splits...")
        save_splits(tokens, "word", OUT_DIR)
        del tokens  # free memory

    # ---- BPE tokenization (for Baseline) ----
    if not args.skip_bpe:
        print(f"\n{'='*60}")
        print("  BPE TOKENIZATION (Baseline)")
        print(f"{'='*60}")

        tokens = tokenize_bpe(CORPUS_PATH, max_bytes)

        print(f"\n  Saving BPE splits...")
        save_splits(tokens, "bpe", OUT_DIR)
        del tokens

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")

    # Print summary of output files
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix in (".bin", ".txt", ".json") and f.name != "corpus.txt":
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
