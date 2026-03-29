"""
Training script for the vanilla Transformer baseline.

Parameter-matched to DisattentionFormer (~27.7M params).
Same corpus, tokenizer, training schedule, and hyperparameters.
Only the architecture differs: standard Transformer vs DisattentionFormer.

This isolates the contribution of:
  - Archetypal curvature tensors (ArchetypalProjection)
  - TensionFFN (opposing-pole feed-forward)
  - IndividuationNorm (depth-decaying normalization)
  - TranscendentFunction (archetype-gated synthesis)
  - Disattention loss (L_constellation, L_tension, L_individuation)
"""

import math
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from desatencao_data import (
    build_dataloader,
    load_cached_corpus,
    WordTokenizer,
    DATA_DIR,
)
from baseline_model import BaselineTransformer

# ---------------------------------------------------------------------------
# Configuration -- matched to DisattentionFormer training
# ---------------------------------------------------------------------------

CONFIG = {
    # model -- parameter-matched to DisattentionFormer (~27.7M)
    "d_model": 272,
    "n_heads": 8,
    "n_layers": 12,
    "d_ff": 1088,
    "seq_len": 512,
    "dropout": 0.1,
    # training -- identical to DisattentionFormer
    "epochs": 10,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "grad_clip": 1.0,
    # logging and saving
    "log_every": 50,
    "save_every": 1000,
    "checkpoint_dir": "checkpoints_baseline",
}


def build_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.exists(checkpoint_dir):
        return None
    pts = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")]
    if not pts:
        return None
    steps = []
    for f in pts:
        try:
            steps.append((int(f.replace("step_", "").replace(".pt", "")), f))
        except ValueError:
            continue
    if not steps:
        return None
    steps.sort()
    return os.path.join(checkpoint_dir, steps[-1][1])


def train():
    config = dict(CONFIG)
    device = get_device()
    print(f"Device: {device}")

    # -- Corpus and DataLoader (same as DisattentionFormer) ------------------
    print("\nPreparing corpus...")
    corpus = load_cached_corpus()

    vocab_path = DATA_DIR / "vocab.txt"
    tokenizer = WordTokenizer()
    if vocab_path.exists():
        tokenizer.load(vocab_path)
        print(f"Loaded vocabulary: {tokenizer.vocab_size:,} words")
    else:
        tokenizer.build(corpus)
        tokenizer.save(vocab_path)
        print(f"Built vocabulary: {tokenizer.vocab_size:,} words (saved to {vocab_path})")
    config["vocab_size"] = tokenizer.vocab_size

    loader = build_dataloader(
        corpus,
        tokenizer,
        seq_len=config["seq_len"],
        batch_size=config["batch_size"],
        num_workers=0,
    )
    print(f"Batches per epoch: {len(loader)}")

    # -- Model ---------------------------------------------------------------
    model = BaselineTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["seq_len"],
        dropout=config["dropout"],
    ).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")

    # -- Optimizer (same groups as DisattentionFormer) ------------------------
    decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    no_decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["lr"],
        betas=(0.9, 0.95),
    )

    total_steps = len(loader) * config["epochs"]
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(config["warmup_steps"], total_steps)
    )

    # -- Checkpoint directory ------------------------------------------------
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # -- Resume from checkpoint if available ---------------------------------
    global_step = 0
    start_epoch = 0
    latest = _find_latest_checkpoint(config["checkpoint_dir"])
    if latest is not None:
        print(f"Resuming from checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        print(f"  Resumed at epoch={start_epoch}, step={global_step}")

    # -- Training loop -------------------------------------------------------
    print(f"\nStarting baseline training: {config['epochs']} epochs, {total_steps} total steps")
    print(
        f"  d_model={config['d_model']}, n_layers={config['n_layers']}, "
        f"n_heads={config['n_heads']}, d_ff={config['d_ff']}"
    )
    print(f"  seq_len={config['seq_len']}, batch_size={config['batch_size']}")
    print(f"  lr={config['lr']}, warmup={config['warmup_steps']}")
    print()

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = model(input_ids)
            loss = F.cross_entropy(
                out["logits"].view(-1, config["vocab_size"]),
                targets.view(-1),
                ignore_index=-1,
            )

            # NaN check before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[NaN/Inf] step={global_step} loss={loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # NaN check on gradients
            nan_detected = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"[NaN/Inf grad] step={global_step} param={name}")
                    nan_detected = True
                    break

            if nan_detected:
                optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            # -- Logging -----------------------------------------------------
            if global_step % config["log_every"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = epoch_steps / max(elapsed, 1e-6)
                print(
                    f"  ep={epoch} step={global_step:>6d} | "
                    f"loss={loss.item():.4f} | "
                    f"lr={lr_now:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            # -- Checkpointing -----------------------------------------------
            if global_step % config["save_every"] == 0:
                ckpt_path = os.path.join(
                    config["checkpoint_dir"], f"step_{global_step}.pt"
                )
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": config,
                    },
                    ckpt_path,
                )
                print(f"  >> Checkpoint saved: {ckpt_path}")

        # -- End of epoch summary --------------------------------------------
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(
            f"\n  Epoch {epoch} complete: avg_loss={avg_loss:.4f}, "
            f"time={elapsed:.0f}s, steps={epoch_steps}\n"
        )

    # -- Save final model ----------------------------------------------------
    final_path = os.path.join(config["checkpoint_dir"], "final.pt")
    torch.save(
        {
            "step": global_step,
            "epoch": config["epochs"],
            "model": model.state_dict(),
            "config": config,
        },
        final_path,
    )
    print(f"Baseline training complete. Final model saved: {final_path}")
    return model


if __name__ == "__main__":
    train()
