"""
Training script for the linguistics-aware JoyceGPT.

Multi-task training with:
- Primary loss: next character prediction
- Auxiliary losses: POS, dependency, morphology, shape prediction

The auxiliary losses are weighted and encourage structured representations
that encode linguistic features in superposition.
"""

import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config, get_device
from dataset import LinguisticDataset
from model import JoyceGPT
from tokenizer import LinguisticTokenizer


def get_lr(
    step: int, warmup_steps: int, max_lr: float, min_lr: float, total_steps: int
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device, eval_batches=50):
    """Estimate loss on train and val sets."""
    model.eval()
    losses = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_char_loss = 0.0
        count = 0
        for i, batch in enumerate(loader):
            if i >= eval_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            result = model(
                idx=batch["x_char"],
                targets=batch["y_char"],
                pos_tags=batch["x_pos"],
                dep_tags=batch["x_dep"],
                morph_tags=batch["x_morph"],
                shape_tags=batch["x_shape"],
                target_pos=batch["y_pos"],
                target_dep=batch["y_dep"],
                target_morph=batch["y_morph"],
                target_shape=batch["y_shape"],
            )
            total_char_loss += result["char_loss"].item()
            count += 1
        losses[split] = total_char_loss / max(count, 1)
    model.train()
    return losses


def save_checkpoint(model, optimizer, step, epoch, best_val_loss, config, path):
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_config": {
                "vocab_size": config.model.vocab_size,
                "d_model": config.model.d_model,
                "n_heads": config.model.n_heads,
                "n_layers": config.model.n_layers,
                "d_ff": config.model.d_ff,
                "max_seq_len": config.model.max_seq_len,
                "dropout": config.model.dropout,
                "bias": config.model.bias,
                "n_pos_tags": config.model.n_pos_tags,
                "n_dep_tags": config.model.n_dep_tags,
                "n_morph_tags": config.model.n_morph_tags,
                "n_shape_tags": config.model.n_shape_tags,
                "d_ling_emb": config.model.d_ling_emb,
            },
        },
        path,
    )


def train() -> None:
    """Main training function."""
    config = Config()
    device = get_device(config.training.device)
    print(f"Using device: {device}")

    # Load tokenizer
    vocab_path = config.data.processed_dir / config.data.vocab_file
    if not vocab_path.exists():
        print("Error: Vocabulary not found. Run prepare_data.py first.")
        sys.exit(1)

    tokenizer = LinguisticTokenizer()
    tokenizer.load(vocab_path)

    # Set vocab sizes in config
    config.model.vocab_size = tokenizer.vocab_sizes["char"]
    config.model.n_pos_tags = tokenizer.vocab_sizes["pos"]
    config.model.n_dep_tags = tokenizer.vocab_sizes["dep"]
    config.model.n_morph_tags = tokenizer.vocab_sizes["morph"]
    config.model.n_shape_tags = tokenizer.vocab_sizes["shape"]
    print(f"Vocab sizes: {tokenizer}")

    # Load data
    train_data = torch.load(config.data.processed_dir / "train.pt", weights_only=False)
    val_data = torch.load(config.data.processed_dir / "val.pt", weights_only=False)

    train_dataset = LinguisticDataset(train_data, config.model.max_seq_len)
    val_dataset = LinguisticDataset(val_data, config.model.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences:   {len(val_dataset):,}")

    # Create model
    model = JoyceGPT.from_config(config.model).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer with weight decay separation
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
    )

    # Resume from checkpoint
    latest_ckpt = config.training.checkpoint_dir / "latest.pt"
    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if latest_ckpt.exists():
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resuming from step {start_step}, epoch {start_epoch}")

    # Training loop
    total_steps = config.training.max_epochs * len(train_loader)
    print(
        f"\nStarting training for {config.training.max_epochs} epochs ({total_steps:,} steps)"
    )
    print(f"Batch size: {config.training.batch_size}")
    print(f"Sequence length: {config.model.max_seq_len}")
    print(
        f"Aux loss weights: pos={config.training.pos_loss_weight}, "
        f"dep={config.training.dep_loss_weight}, "
        f"morph={config.training.morph_loss_weight}, "
        f"shape={config.training.shape_loss_weight}"
    )
    print("-" * 70)

    model.train()
    global_step = start_step
    t0 = time.time()

    for epoch in range(start_epoch, config.training.max_epochs):
        for batch_idx, batch in enumerate(train_loader):
            if global_step < start_step:
                global_step += 1
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            # Learning rate schedule
            lr = get_lr(
                global_step,
                config.training.warmup_steps,
                config.training.learning_rate,
                config.training.min_lr,
                total_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            result = model(
                idx=batch["x_char"],
                targets=batch["y_char"],
                pos_tags=batch["x_pos"],
                dep_tags=batch["x_dep"],
                morph_tags=batch["x_morph"],
                shape_tags=batch["x_shape"],
                target_pos=batch["y_pos"],
                target_dep=batch["y_dep"],
                target_morph=batch["y_morph"],
                target_shape=batch["y_shape"],
            )

            # Compute total loss: char_loss + weighted auxiliary losses
            loss = result["char_loss"]
            if "aux_losses" in result:
                loss = (
                    loss + config.training.pos_loss_weight * result["aux_losses"]["pos"]
                )
                loss = (
                    loss + config.training.dep_loss_weight * result["aux_losses"]["dep"]
                )
                loss = (
                    loss
                    + config.training.morph_loss_weight * result["aux_losses"]["morph"]
                )
                loss = (
                    loss
                    + config.training.shape_loss_weight * result["aux_losses"]["shape"]
                )

            loss.backward()

            if config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            if global_step % config.training.log_interval == 0:
                dt = time.time() - t0
                chars_per_sec = (
                    config.training.batch_size
                    * config.model.max_seq_len
                    * config.training.log_interval
                    / max(dt, 1e-6)
                )
                aux_str = ""
                if "aux_losses" in result:
                    aux_str = (
                        f" | pos {result['aux_losses']['pos'].item():.3f}"
                        f" dep {result['aux_losses']['dep'].item():.3f}"
                        f" morph {result['aux_losses']['morph'].item():.3f}"
                        f" shape {result['aux_losses']['shape'].item():.3f}"
                    )
                print(
                    f"step {global_step:>6d} | epoch {epoch:>3d} | "
                    f"char_loss {result['char_loss'].item():.4f} | "
                    f"total {loss.item():.4f} | lr {lr:.2e} | "
                    f"{chars_per_sec:.0f} ch/s{aux_str}"
                )
                t0 = time.time()

            # Evaluation
            if global_step > 0 and global_step % config.training.eval_interval == 0:
                losses = estimate_loss(model, train_loader, val_loader, device)
                print(
                    f"  eval | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}"
                )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(
                        model,
                        optimizer,
                        global_step,
                        epoch,
                        best_val_loss,
                        config,
                        config.training.checkpoint_dir / "best.pt",
                    )
                    print(f"  New best val_loss: {best_val_loss:.4f} (saved)")

            # Periodic checkpoint
            if global_step > 0 and global_step % config.training.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    epoch,
                    best_val_loss,
                    config,
                    latest_ckpt,
                )

            global_step += 1

        # End of epoch checkpoint
        save_checkpoint(
            model, optimizer, global_step, epoch + 1, best_val_loss, config, latest_ckpt
        )
        print(f"Epoch {epoch} complete. Saved checkpoint.")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
