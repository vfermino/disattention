"""
Gutenberg dataset and dataloader for 85M-parameter training.

Supports both word-level (V1/V2) and BPE (Baseline) tokenization.
Uses memory-mapped files for memory efficiency with large corpora.

The word-level tokenizer is compatible with desatencao_data.WordTokenizer.
"""

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/gutenberg")

# Special tokens (same as desatencao_data.py)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOT_TOKEN = "<eot>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOT_TOKEN]


def tokenize_words(text: str) -> list[str]:
    """Split text into word tokens and punctuation (same as desatencao_data.py)."""
    return re.findall(r"[A-Za-z'\u2019-]+|[0-9]+|[^\s]", text)


# ---------------------------------------------------------------------------
# Word-level tokenizer (compatible with desatencao_data.WordTokenizer)
# ---------------------------------------------------------------------------

class GutenbergWordTokenizer:
    """Word-level tokenizer loaded from pre-built vocab file."""

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def load(self, path: Path) -> "GutenbergWordTokenizer":
        """Load vocab from text file (one word per line)."""
        self.idx2word = path.read_text(encoding="utf-8").split("\n")
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)

    @property
    def pad_id(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @property
    def eot_id(self) -> int:
        return self.word2idx[EOT_TOKEN]

    def encode(self, text: str) -> list[int]:
        return [self.word2idx.get(w, self.unk_id) for w in tokenize_words(text)]

    def decode(self, ids: list[int]) -> str:
        words = [self.idx2word[i] if i < len(self.idx2word) else UNK_TOKEN for i in ids]
        parts = []
        for w in words:
            if w in SPECIAL_TOKENS:
                continue
            if parts and len(w) == 1 and not w.isalnum():
                parts.append(w)
            else:
                if parts:
                    parts.append(" ")
                parts.append(w)
        return "".join(parts)


# ---------------------------------------------------------------------------
# BPE tokenizer wrapper (for Baseline)
# ---------------------------------------------------------------------------

class GutenbergBPETokenizer:
    """Wrapper around tiktoken GPT-2 BPE for uniform interface."""

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return 50257  # GPT-2 BPE vocab size

    @property
    def pad_id(self) -> int:
        return self.vocab_size - 1  # use last token as pad

    @property
    def eot_id(self) -> int:
        return self._enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


# ---------------------------------------------------------------------------
# Memory-mapped dataset
# ---------------------------------------------------------------------------

class MemmapDataset(Dataset):
    """
    Dataset backed by a memory-mapped .bin file of int32 token IDs.

    Chunks the token stream into fixed-length sequences for next-token
    prediction. The order of the corpus is preserved (no shuffling).
    """

    def __init__(self, bin_path: Path, seq_len: int = 512):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.int32, mode="r")
        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start: start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def build_gutenberg_dataloader(
    split: str = "train",
    tokenizer_type: str = "word",
    seq_len: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    data_dir: Path = DATA_DIR,
) -> DataLoader:
    """
    Build a DataLoader for Gutenberg training.

    Args:
        split: "train" or "val"
        tokenizer_type: "word" (V1/V2) or "bpe" (Baseline)
        seq_len: sequence length
        batch_size: batch size
        num_workers: DataLoader workers
        data_dir: path to data/gutenberg/

    Returns:
        DataLoader yielding (input_ids, targets) tuples
    """
    bin_path = data_dir / f"{tokenizer_type}_{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"{bin_path} not found. Run gutenberg_prepare.py first."
        )

    ds = MemmapDataset(bin_path, seq_len)
    print(f"  [{tokenizer_type}/{split}] {len(ds):,} sequences "
          f"(seq_len={seq_len}, {bin_path.stat().st_size / 1e6:.0f}MB)")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # preserve corpus order
        num_workers=num_workers,
        pin_memory=False,  # MPS does not support pin_memory
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Convenience: load tokenizer
# ---------------------------------------------------------------------------

def load_word_tokenizer(data_dir: Path = DATA_DIR) -> GutenbergWordTokenizer:
    """Load pre-built word-level tokenizer."""
    vocab_path = data_dir / "word_vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"{vocab_path} not found. Run gutenberg_prepare.py first."
        )
    tok = GutenbergWordTokenizer().load(vocab_path)
    print(f"  Word tokenizer: {tok.vocab_size:,} tokens")
    return tok


def load_bpe_tokenizer() -> GutenbergBPETokenizer:
    """Load BPE tokenizer (tiktoken GPT-2)."""
    tok = GutenbergBPETokenizer()
    print(f"  BPE tokenizer: {tok.vocab_size:,} tokens")
    return tok
