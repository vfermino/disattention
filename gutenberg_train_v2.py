"""
Training loop for DisattentionFormer V2 (Titans) on Gutenberg corpus (~85M params).

Word-level tokenization — part of the Disattention architecture.
Uses memory-mapped dataset for efficient large-corpus training.

Non-destructive: does NOT touch existing mythic checkpoints.
Outputs to checkpoints_gutenberg_v2/.
"""

import math
import os
import time

import torch
import torch.optim as optim

from build_archetypes import ARCHETYPES, build_archetype_tensors
from desatencao_loss import disattention_loss
from disattention_v2_model import DisattentionFormerV2
from gutenberg_data import (
    build_gutenberg_dataloader,
    load_word_tokenizer,
)

# ---------------------------------------------------------------------------
# Configuration — 85M params on Gutenberg (word-level, Titans memory)
# ---------------------------------------------------------------------------

CONFIG = {
    # model
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 7,            # 84.8M params (fewer layers — memory handles long-range)
    "d_ff": 2048,
    "seq_len": 512,
    "dropout": 0.1,
    "chunk_size": 4,          # temporal granularity for neural memory
    "memory_depth": 2,        # depth of memory MLP
    # training
    "epochs": 3,
    "batch_size": 4,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 4000,
    "grad_clip": 1.0,
    # loss weights
    "alpha": 0.10,
    "beta": 0.05,
    "gamma": 0.02,
    # logging and saving
    "log_every": 100,
    "save_every": 5000,
    "checkpoint_dir": "checkpoints_gutenberg_v2",
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = [f for f in os.listdir(checkpoint_dir)
             if f.startswith("step_") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f.replace("step_", "").replace(".pt", "")))
    return os.path.join(checkpoint_dir, ckpts[-1])


def train(config: dict):
    device = get_device()
    print(f"Device: {device}")

    # -- Archetype tensors (rebuilt at d_model=512) -------------------------
    tensor_path = f"archetype_tensors_{config['d_model']}.pt"
    if os.path.exists(tensor_path):
        print(f"Loading archetype tensors from {tensor_path}")
        archetype_tensors = torch.load(tensor_path, map_location=device, weights_only=True)
    else:
        print(f"Building archetype tensors (d={config['d_model']})...")
        archetype_tensors = build_archetype_tensors(
            ARCHETYPES, target_dim=config["d_model"], device="cpu"
        )
        torch.save(archetype_tensors, tensor_path)
        print(f"Saved: {tensor_path}")
    archetype_tensors = {k: v.to(device) for k, v in archetype_tensors.items()}

    # -- Tokenizer and DataLoader -------------------------------------------
    print("\nLoading Gutenberg word tokenizer...")
    tokenizer = load_word_tokenizer()
    config["vocab_size"] = tokenizer.vocab_size

    print("\nBuilding dataloaders...")
    train_loader = build_gutenberg_dataloader(
        split="train", tokenizer_type="word",
        seq_len=config["seq_len"], batch_size=config["batch_size"],
    )
    val_loader = build_gutenberg_dataloader(
        split="val", tokenizer_type="word",
        seq_len=config["seq_len"], batch_size=config["batch_size"],
    )
    print(f"Train batches per epoch: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    # -- Model --------------------------------------------------------------
    model = DisattentionFormerV2(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["seq_len"],
        archetype_tensors=archetype_tensors,
        chunk_size=config["chunk_size"],
        memory_depth=config["memory_depth"],
        dropout=config["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # -- Optimizer ----------------------------------------------------------
    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["lr"],
        betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * config["epochs"]
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(config["warmup_steps"], total_steps)
    )

    # -- Checkpoint resume --------------------------------------------------
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    archetype_names = list(archetype_tensors.keys())

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
    print(f"\nStarting training: {config['epochs']} epochs, {total_steps:,} total steps")
    print(f"  d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"n_heads={config['n_heads']}, d_ff={config['d_ff']}")
    print(f"  vocab_size={config['vocab_size']:,}, seq_len={config['seq_len']}")
    print(f"  chunk_size={config['chunk_size']}, memory_depth={config['memory_depth']}")
    print(f"  batch_size={config['batch_size']}, lr={config['lr']}")
    print()

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

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

            if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                print(f"[NaN/Inf] step={global_step}")
                optimizer.zero_grad(set_to_none=True)
                continue

            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += losses["loss"].item()
            epoch_steps += 1

            if global_step == 1:
                print(f"  Step 1 OK: loss={losses['loss'].item():.4f}", flush=True)

            if global_step % config["log_every"] == 0:
                w_mean = out["archetype_weights"].detach().mean(dim=0)
                dominant = archetype_names[w_mean.argmax().item()]
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_s = epoch_steps / max(elapsed, 1e-6)

                print(f"  ep={epoch} step={global_step:>7d} | "
                      f"loss={losses['loss'].item():.4f} "
                      f"ce={losses['l_ce'].item():.4f} | "
                      f"arch={dominant} | lr={lr_now:.2e} | {steps_s:.1f} steps/s",
                      flush=True)

            if global_step % config["save_every"] == 0:
                ckpt_path = os.path.join(config["checkpoint_dir"], f"step_{global_step}.pt")
                torch.save({
                    "step": global_step, "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                }, ckpt_path)
                print(f"  >> Checkpoint: {ckpt_path}")

        # -- End of epoch ---------------------------------------------------
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch} complete: avg_loss={avg_loss:.4f}, "
              f"time={elapsed:.0f}s, steps={epoch_steps}")

        # -- Validation -----------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                out = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    out["logits"].view(-1, config["vocab_size"]),
                    targets.view(-1),
                )
                val_loss += loss.item()
                val_steps += 1
        print(f"  Val CE: {val_loss / max(val_steps, 1):.4f}\n")

    # -- Save final ---------------------------------------------------------
    final_path = os.path.join(config["checkpoint_dir"], "final.pt")
    torch.save({
        "step": global_step, "epoch": config["epochs"],
        "model": model.state_dict(), "config": config,
    }, final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    train(CONFIG)
