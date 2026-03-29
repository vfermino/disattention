"""Quick smoke test to see if the 85M V1 model runs on MPS."""
import time
import torch
from build_archetypes import ARCHETYPES, build_archetype_tensors
from desatencao_model import DisattentionFormer
from desatencao_loss import disattention_loss

device = torch.device("mps")
print("Loading archetype tensors...")
at = torch.load("archetype_tensors_512.pt", map_location=device, weights_only=True)

print("Creating model...")
model = DisattentionFormer(
    vocab_size=80003, d_model=512, n_heads=8, n_layers=8, d_ff=2048,
    max_seq_len=512, archetype_tensors=at, dropout=0.1,
).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n:,}")

print("Forward pass (batch=4, seq=512)...")
x = torch.randint(0, 80003, (4, 512), device=device)
t = torch.randint(0, 80003, (4, 512), device=device)

t0 = time.time()
out = model(x, return_internals=True)
torch.mps.synchronize()
print(f"  Forward: {time.time()-t0:.2f}s")
print(f"  Logits: {out['logits'].shape}")
print(f"  Arch weights: {out['archetype_weights'].shape}")

print("Loss...")
t0 = time.time()
losses = disattention_loss(
    logits=out["logits"], targets=t, model=model,
    x_prefnn=out["x_prefnn"], archetype_weights=out["archetype_weights"],
    alpha=0.1, beta=0.05, gamma=0.02,
)
print(f"  Loss compute: {time.time()-t0:.2f}s, loss={losses['loss'].item():.4f}")

print("Backward...")
t0 = time.time()
losses["loss"].backward()
torch.mps.synchronize()
print(f"  Backward: {time.time()-t0:.2f}s")

print("\nNow timing 5 full steps (batch=8)...")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(5):
    x = torch.randint(0, 80003, (8, 512), device=device)
    t = torch.randint(0, 80003, (8, 512), device=device)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    out = model(x, return_internals=True)
    losses = disattention_loss(
        logits=out["logits"], targets=t, model=model,
        x_prefnn=out["x_prefnn"], archetype_weights=out["archetype_weights"],
        alpha=0.1, beta=0.05, gamma=0.02,
    )
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.mps.synchronize()
    elapsed = time.time() - t0
    print(f"  Step {i}: {elapsed:.2f}s, loss={losses['loss'].item():.4f}")

print("\nDone. Training is feasible.")
