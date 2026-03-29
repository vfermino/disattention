"""
Multi-layer tokenizer for the Joyce linguistics-aware corpus.

Handles:
- Character-level tokenization (text)
- POS tag tokenization
- Dependency relation tokenization
- Morphological feature tokenization
- Word shape tokenization

Each layer has its own vocabulary mapping. All are aligned at the
character level: every character inherits the linguistic features
of the word/token it belongs to.
"""

from __future__ import annotations

import json
from pathlib import Path


class LinguisticTokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    # The linguistic feature layers we track
    LAYERS = ("char", "pos", "dep", "morph", "shape")

    def __init__(self):
        # Each layer gets its own vocabulary
        self.vocabs: dict[str, dict[str, int]] = {layer: {} for layer in self.LAYERS}
        self.reverse_vocabs: dict[str, dict[int, str]] = {
            layer: {} for layer in self.LAYERS
        }
        self.vocab_sizes: dict[str, int] = {layer: 0 for layer in self.LAYERS}

    def build_vocab(self, layer: str, values: list[str]) -> None:
        """Build vocabulary for a specific layer from observed values."""
        unique = sorted(set(values))
        vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for i, v in enumerate(unique, start=2):
            vocab[v] = i
        self.vocabs[layer] = vocab
        self.reverse_vocabs[layer] = {idx: val for val, idx in vocab.items()}
        self.vocab_sizes[layer] = len(vocab)

    def encode_layer(self, layer: str, values: list[str]) -> list[int]:
        """Encode a list of values for a given layer."""
        vocab = self.vocabs[layer]
        unk_idx = vocab[self.UNK_TOKEN]
        return [vocab.get(v, unk_idx) for v in values]

    def decode_layer(self, layer: str, indices: list[int]) -> list[str]:
        """Decode a list of indices for a given layer."""
        rev = self.reverse_vocabs[layer]
        pad_idx = self.vocabs[layer][self.PAD_TOKEN]
        return [rev.get(idx, "?") for idx in indices if idx != pad_idx]

    def decode_chars(self, indices: list[int]) -> str:
        """Decode character indices back to a string."""
        rev = self.reverse_vocabs["char"]
        pad_idx = self.vocabs["char"][self.PAD_TOKEN]
        unk_idx = self.vocabs["char"][self.UNK_TOKEN]
        chars = []
        for idx in indices:
            if idx == pad_idx:
                continue
            if idx == unk_idx:
                chars.append("?")
            else:
                chars.append(rev.get(idx, "?"))
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Save all vocabularies to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocabs": self.vocabs,
            "vocab_sizes": self.vocab_sizes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        """Load all vocabularies from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocabs = data["vocabs"]
        self.vocab_sizes = data["vocab_sizes"]
        self.reverse_vocabs = {}
        for layer in self.LAYERS:
            self.reverse_vocabs[layer] = {
                int(idx): val for val, idx in self.vocabs[layer].items()
            }

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k}={v}" for k, v in self.vocab_sizes.items())
        return f"LinguisticTokenizer({sizes})"
