"""
Training loop for the DisattentionFormer.

Adapted for Apple Silicon M3 (MPS backend):
  - No AMP (MPS does not support torch.cuda.amp)
  - float32 throughout
  - Cosine LR with linear warmup
  - Gradient clipping
  - Checkpointing with full state
  - Archetype diagnostics every log_every steps

shuffle=False -- the corpus order is the architecture.
"""

import math
import os
import time

import torch
import torch.optim as optim

from build_archetypes import ARCHETYPES, build_archetype_tensors
from desatencao_data import (
    build_dataloader,
    load_cached_corpus,
    WordTokenizer,
    DATA_DIR,
)
from desatencao_loss import disattention_loss
from desatencao_model import DisattentionFormer

# ---------------------------------------------------------------------------
# Configuration -- scaled for MacBook Pro M3 18 GB
# ---------------------------------------------------------------------------

CONFIG = {
    # model (vocab_size determined at runtime from corpus)
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 8,
    "d_ff": 1024,
    "seq_len": 512,
    "dropout": 0.1,
    # training
    "epochs": 10,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "grad_clip": 1.0,
    # loss weights
    "alpha": 0.10,  # L_constellation
    "beta": 0.05,  # L_tension (maximized via negation)
    "gamma": 0.02,  # L_individuation (maximized via negation)
    # logging and saving
    "log_every": 50,
    "save_every": 1000,
    "checkpoint_dir": "checkpoints",
    "use_wandb": False,
}


def get_device() -> torch.device:
    """Select the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay to zero."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def train(config: dict):
    device = get_device()
    print(f"Device: {device}")

    # -- Archetype tensors --------------------------------------------------
    tensor_path = "archetype_tensors.pt"
    if os.path.exists(tensor_path):
        print(f"Loading archetype tensors from {tensor_path}")
        archetype_tensors = torch.load(
            tensor_path, map_location=device, weights_only=True
        )
    else:
        print("Building archetype tensors (first run)...")
        archetype_tensors = build_archetype_tensors(
            ARCHETYPES, target_dim=config["d_model"], device="cpu"
        )
        torch.save(archetype_tensors, tensor_path)
        print(f"Saved: {tensor_path}")
    archetype_tensors = {k: v.to(device) for k, v in archetype_tensors.items()}

    # -- Corpus and DataLoader ----------------------------------------------
    print("\nPreparing corpus...")
    corpus = load_cached_corpus()

    # Build word-level tokenizer from corpus
    vocab_path = DATA_DIR / "vocab.txt"
    tokenizer = WordTokenizer()
    if vocab_path.exists():
        tokenizer.load(vocab_path)
        print(f"Loaded vocabulary: {tokenizer.vocab_size:,} words")
    else:
        tokenizer.build(corpus)
        tokenizer.save(vocab_path)
        print(
            f"Built vocabulary: {tokenizer.vocab_size:,} words (saved to {vocab_path})"
        )
    config["vocab_size"] = tokenizer.vocab_size

    loader = build_dataloader(
        corpus,
        tokenizer,
        seq_len=config["seq_len"],
        batch_size=config["batch_size"],
        num_workers=0,
    )
    print(f"Batches per epoch: {len(loader)}")

    # -- Model --------------------------------------------------------------
    model = DisattentionFormer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["seq_len"],
        archetype_tensors=archetype_tensors,
        dropout=config["dropout"],
    ).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")

    # -- Optimizer: separate decay / no-decay groups ------------------------
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

    # -- Optional wandb -----------------------------------------------------
    if config["use_wandb"]:
        try:
            import wandb

            wandb.init(project="disattentionformer", config=config)
        except ImportError:
            print("wandb not installed -- logging to stdout only")
            config["use_wandb"] = False

    # -- Checkpoint directory -----------------------------------------------
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    archetype_names = list(archetype_tensors.keys())

    # -- Resume from checkpoint if available --------------------------------
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

    # -- Training loop ------------------------------------------------------
    print(f"\nStarting training: {config['epochs']} epochs, {total_steps} total steps")
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

            # Forward pass -- no AMP on MPS, plain float32
            out = model(input_ids, return_internals=True)
            losses = disattention_loss(
                logits=out["logits"],
                targets=targets,
                model=model,
                x_prefnn=out["x_prefnn"],
                archetype_weights=out["archetype_weights"],
                alpha=config["alpha"],
                beta=config["beta"],
                gamma=config["gamma"],
            )

            # Check for NaN in loss before backward (prevents NaN propagation)
            if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                print(
                    f"[NaN/Inf] step={global_step} | "
                    f"loss={losses['loss'].item()} "
                    f"ce={losses['l_ce'].item():.4f} "
                    f"const={losses['l_const'].item():.4f} "
                    f"tens={losses['l_tension'].item():.4f} "
                    f"indiv={losses['l_individ'].item():.4f}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            losses["loss"].backward()

            # Check gradients for NaN after backward
            nan_detected = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"[NaN/Inf grad] step={global_step} param={name}")
                    nan_detected = True
                    break

            if nan_detected:
                print(f"[NaN] Aborting step {global_step}, clearing gradients")
                optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += losses["loss"].item()
            epoch_steps += 1

            # -- Logging ----------------------------------------------------
            if global_step % config["log_every"] == 0:
                w_mean = out["archetype_weights"].detach().mean(dim=0)
                dominant_idx = w_mean.argmax().item()
                dominant = archetype_names[dominant_idx]
                w_list = w_mean.cpu().tolist()

                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = epoch_steps / max(elapsed, 1e-6)

                print(
                    f"  ep={epoch} step={global_step:>6d} | "
                    f"loss={losses['loss'].item():.4f} "
                    f"ce={losses['l_ce'].item():.4f} "
                    f"const={losses['l_const'].item():.4f} "
                    f"tens={losses['l_tension'].item():.4f} "
                    f"indiv={losses['l_individ'].item():.4f} | "
                    f"arch={dominant} | "
                    f"lr={lr_now:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                # Print full archetype weight vector every 200 steps
                if global_step % (config["log_every"] * 4) == 0:
                    w_str = " ".join(
                        f"{n}={v:.3f}" for n, v in zip(archetype_names, w_list)
                    )
                    print(f"    archetypes: {w_str}")

                if config["use_wandb"]:
                    import wandb

                    log = {
                        "loss/total": losses["loss"].item(),
                        "loss/ce": losses["l_ce"].item(),
                        "loss/constellation": losses["l_const"].item(),
                        "loss/tension": losses["l_tension"].item(),
                        "loss/individuation": losses["l_individ"].item(),
                        "archetype/dominant": dominant,
                        "lr": lr_now,
                        "epoch": epoch,
                        "step": global_step,
                    }
                    for n, v in zip(archetype_names, w_list):
                        log[f"archetype/{n}"] = v
                    wandb.log(log)

            # -- Checkpointing ----------------------------------------------
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

        # -- End of epoch summary -------------------------------------------
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(
            f"\n  Epoch {epoch} complete: avg_loss={avg_loss:.4f}, "
            f"time={elapsed:.0f}s, steps={epoch_steps}\n"
        )

    # -- Save final model ---------------------------------------------------
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
    print(f"Training complete. Final model saved: {final_path}")
    return model


def _find_latest_checkpoint(checkpoint_dir: str):
    """Find the most recent step_*.pt checkpoint."""
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ]
    if not ckpts:
        return None
    # Sort by step number
    ckpts.sort(key=lambda f: int(f.replace("step_", "").replace(".pt", "")))
    return os.path.join(checkpoint_dir, ckpts[-1])


if __name__ == "__main__":
    train(CONFIG)
