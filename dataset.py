"""
Multi-layer dataset for linguistics-aware character-level text generation.

Each sample contains parallel sequences for:
- Character indices (primary task)
- POS tag indices
- Dependency relation indices
- Morphological feature indices
- Word shape indices
"""

import torch
from torch.utils.data import Dataset


class LinguisticDataset(Dataset):
    """
    Dataset that produces aligned multi-layer sequences.

    Each item returns:
        x_char, y_char: input/target character sequences
        x_pos, y_pos: input/target POS tag sequences
        x_dep, y_dep: input/target dependency relation sequences
        x_morph, y_morph: input/target morphological feature sequences
        x_shape, y_shape: input/target word shape sequences
    """

    def __init__(self, data: dict, seq_len: int):
        """
        Args:
            data: Dict with keys 'char', 'pos', 'dep', 'morph', 'shape',
                  each mapping to a list of token indices.
            seq_len: Length of each training sequence.
        """
        self.seq_len = seq_len
        self.layers = {}
        for key in ("char", "pos", "dep", "morph", "shape"):
            self.layers[key] = torch.tensor(data[key], dtype=torch.long)

        self.length = max(0, len(self.layers["char"]) - self.seq_len)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        result = {}
        for key, tensor in self.layers.items():
            result[f"x_{key}"] = tensor[idx : idx + self.seq_len]
            result[f"y_{key}"] = tensor[idx + 1 : idx + self.seq_len + 1]
        return result
