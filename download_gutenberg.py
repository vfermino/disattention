"""
Download manu/project_gutenberg from HuggingFace and save to data/gutenberg/.
Large dataset — streams to disk to avoid OOM.
"""
import os
import sys
import json
from datasets import load_dataset

OUT_DIR = "data/gutenberg"
os.makedirs(OUT_DIR, exist_ok=True)

# Available splits: de, en, es, fr, it, nl, pl, pt, ru, sv, zh
LANG = "en"
print(f"Loading manu/project_gutenberg split='{LANG}' (streaming)...", flush=True)
ds = load_dataset("manu/project_gutenberg", split=LANG, streaming=True)
print("Dataset stream ready.", flush=True)

# Stream and save texts as individual files + metadata
metadata = []
total_chars = 0
count = 0

# Write a single concatenated corpus file for training,
# plus individual files for inspection
corpus_path = os.path.join(OUT_DIR, "corpus.txt")

with open(corpus_path, "w", encoding="utf-8") as corpus_f:
    for i, example in enumerate(ds):
        # Inspect first example to find column names
        if i == 0:
            print(f"Dataset columns: {list(example.keys())}")
            # Find the text column
            text_col = None
            for candidate in ["text", "Text", "content", "body", "document"]:
                if candidate in example:
                    text_col = candidate
                    break
            if text_col is None:
                # Use the first string column
                for k, v in example.items():
                    if isinstance(v, str) and len(v) > 100:
                        text_col = k
                        break
            if text_col is None:
                print(f"Could not find text column! Keys: {list(example.keys())}")
                print(f"First example: {example}")
                break
            print(f"Using text column: '{text_col}'", flush=True)

        text = example.get(text_col, "")
        if not text or len(text.strip()) < 50:
            continue

        # Write to corpus
        corpus_f.write(text.strip())
        corpus_f.write("\n\n")

        total_chars += len(text)
        count += 1

        # Track metadata (only keep first 500 for the JSON)
        if len(metadata) < 500:
            meta = {k: v for k, v in example.items() if k != text_col and not isinstance(v, (bytes, bytearray))}
            meta["char_count"] = len(text)
            metadata.append(meta)

        if count % 500 == 0:
            print(f"  {count} texts downloaded, {total_chars / 1e6:.1f}M chars...", flush=True)

print(f"\nDone! {count} texts, {total_chars / 1e6:.1f}M chars total")
print(f"Corpus saved to: {corpus_path}")

# Save metadata
meta_path = os.path.join(OUT_DIR, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({"total_texts": count, "total_chars": total_chars, "texts": metadata[:100]}, f, indent=2, default=str)
print(f"Metadata saved to: {meta_path}")
