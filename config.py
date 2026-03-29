"""
Configuration for the Joyce linguistics-aware character-level Transformer.

Informed by "Superposition Yields Robust Neural Scaling" (Liu et al., 2025):
the model jointly predicts characters and linguistic features (POS, dependency,
morphology) to encourage structured superposition of power-law-distributed
linguistic features in the model's representations.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    corpus_file: str = "joyce_corpus.txt"
    vocab_file: str = "vocab.json"
    train_split: float = 0.9

    # Project Gutenberg URLs for Joyce's works
    sources: dict = field(
        default_factory=lambda: {
            "dubliners": "https://www.gutenberg.org/cache/epub/2814/pg2814.txt",
            "portrait": "https://www.gutenberg.org/cache/epub/4217/pg4217.txt",
            "ulysses": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
            "chamber_music": "https://www.gutenberg.org/cache/epub/2817/pg2817.txt",
            "exiles": "https://www.gutenberg.org/cache/epub/55945/pg55945.txt",
        }
    )


@dataclass
class ModelConfig:
    # Vocab sizes are set dynamically after building vocabularies
    vocab_size: int = 0  # character vocabulary
    n_pos_tags: int = 0  # POS tag vocabulary
    n_dep_tags: int = 0  # dependency relation vocabulary
    n_morph_tags: int = 0  # morphological feature vocabulary
    n_shape_tags: int = 0  # word shape vocabulary

    # Transformer architecture
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536  # 4 * d_model
    dropout: float = 0.1
    max_seq_len: int = 512
    bias: bool = False

    # Linguistic feature embedding dimension (added to char embedding)
    d_ling_emb: int = 64  # dimension for each linguistic feature embedding


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 50
    grad_clip: float = 1.0

    # Scheduler
    warmup_steps: int = 200
    min_lr: float = 3e-5

    # Auxiliary loss weights for linguistic features
    # These encourage structured representations per the superposition paper
    pos_loss_weight: float = 0.3
    dep_loss_weight: float = 0.2
    morph_loss_weight: float = 0.15
    shape_loss_weight: float = 0.1

    # Logging and checkpointing
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: Path = Path("checkpoints")

    # Device
    device: str = "auto"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1000
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def get_device(preference: str = "auto") -> str:
    """Resolve device preference to an actual device string."""
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return preference
