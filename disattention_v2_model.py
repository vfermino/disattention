

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Neural Memory (Titans backbone, following Aedelon/Titans-PyTorch-MLX)
# ---------------------------------------------------------------------------


class MemoryMLP(nn.Module):
    """
    MLP serving as neural memory (Aedelon-style with nn.Linear layers).

    The weights of this network ARE the memory. They are updated at each
    time step via surprise (gradient of associative memory loss).

    Using nn.Linear (no bias) so torch.autograd.grad works cleanly.
    For L_M = 1: linear memory (equivalent to matrix-valued fast weights).
    For L_M >= 2: deep memory with SiLU activations.
    """

    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(dim, dim, bias=False))

        self.activation = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        return [layer.weight.data.clone() for layer in self.layers]

    def set_weights(self, weights: list[torch.Tensor]):
        for layer, w in zip(self.layers, weights):
            layer.weight.data.copy_(w)

    def compute_loss(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Associative memory loss: ||M(k) - v||^2 (Eq. 12)."""
        predictions = self.forward(keys)
        return F.mse_loss(predictions, values, reduction="mean")


class ArchetypalNeuralMemory(nn.Module):
    """
    Titans-style neural memory with Jungian archetypal modulation.
    Follows the Aedelon implementation for MPS compatibility.

    Key design (paper Section 3.1 + our extension):
      - Memory update:  M_t = (1 - α_t) · M_{t-1} + S_t       (Eq. 13)
      - Surprise:       S_t = η_t · S_{t-1} - θ_t · ∇ℓ        (Eq. 14)
      - Loss:           ℓ(M; x_t) = ||M(k_t) - v_t||^2        (Eq. 12)

    Our extension: keys are projected through the archetypal metric M
    before storage, so the memory operates in curved space. Surprise
    is measured in archetypally deformed geometry.

    Gradient computation uses torch.autograd.grad (Aedelon approach)
    instead of manual backward — works on MPS, handles any MLP depth.
    Queries and keys are L2-normalized and SiLU-activated per Section 4.4.
    """

    def __init__(
        self,
        dim: int,
        chunk_size: int = 4,
        memory_depth: int = 2,
        memory_lr: float = 0.1,
        memory_momentum: float = 0.9,
        memory_decay: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.memory_lr = memory_lr
        self.memory_momentum = memory_momentum
        self.memory_decay = memory_decay

        # The memory model (weights = the memory itself)
        self.memory = MemoryMLP(dim, memory_depth)

        # Projections for keys, values, queries
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_q = nn.Linear(dim, dim, bias=False)

        # Output projection (after retrieval)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        # Data-dependent gates: α_t (decay), θ_t (lr), η_t (momentum)
        # Computed from chunk context, producing per-chunk scalars
        self.gate_decay = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_lr = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_momentum = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        # Normalization
        self.store_norm = nn.RMSNorm(dim)
        self.retrieve_norm = nn.RMSNorm(dim)

        # Init projections
        for m in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.xavier_uniform_(m.weight)

    def _compute_gradients(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Compute gradients of the associative memory loss w.r.t. memory weights.

        Uses torch.autograd.grad (Aedelon approach) — works on MPS,
        handles any MLP depth, no manual backward needed.

        Args:
            keys:   [B, S, dim] — what to store under
            values: [B, S, dim] — what to store

        Returns:
            List of gradient tensors, one per memory layer weight.
        """
        with torch.enable_grad():
            for param in self.memory.parameters():
                param.requires_grad_(True)

            keys_d = keys.detach().requires_grad_(False)
            values_d = values.detach()

            loss = self.memory.compute_loss(keys_d, values_d)

            grads = torch.autograd.grad(
                loss,
                list(self.memory.parameters()),
                create_graph=False,
                allow_unused=True,
            )

            for param in self.memory.parameters():
                param.requires_grad_(False)

        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.memory.parameters())
        ]

    def _init_state(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initialize memory state: (weights, momentum)."""
        weights = self.memory.get_weights()
        momentum = [torch.zeros_like(w) for w in weights]
        return weights, momentum

    def _update_memory(
        self,
        weights: list[torch.Tensor],
        momentum: list[torch.Tensor],
        grads: list[torch.Tensor],
        alpha: float,
        eta: float,
        theta: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Paper-correct memory update (Eq. 13-14):
            S_t = η · S_{t-1} - θ · ∇ℓ         (surprise with momentum)
            M_t = (1 - α) · M_{t-1} + S_t       (update with forgetting)
        """
        new_momentum = []
        for m, g in zip(momentum, grads):
            s = eta * m - theta * g
            new_momentum.append(s)

        new_weights = []
        for w, s in zip(weights, new_momentum):
            w_new = (1.0 - alpha) * w + s
            new_weights.append(w_new)

        return new_weights, new_momentum

    def forward(
        self,
        x: torch.Tensor,       # [B, S, dim]
        M: torch.Tensor,       # [B, dim, dim] — archetypal metric
    ) -> torch.Tensor:
        """
        Process sequence through neural memory.

        For each chunk of the sequence:
          1. Compute data-dependent gates (α, θ, η) from chunk context
          2. Project keys through archetypal metric M (curved space)
          3. Compute surprise = gradient of ||M(k) - v||^2
          4. Update memory weights with momentum and decay
          5. Retrieve from updated memory using queries

        The memory processes chunks causally: chunk t is retrieved using
        weights updated by chunks 0..t-1.

        Args:
            x:  input embeddings [B, S, dim]
            M:  archetypal curvature metric [B, dim, dim]

        Returns:
            memory_out: [B, S, dim]
        """
        B, S, D = x.shape

        # --- Pad sequence to multiple of chunk_size ---
        pad = (self.chunk_size - S % self.chunk_size) % self.chunk_size
        if pad > 0:
            x_padded = F.pad(x, (0, 0, 0, pad))
        else:
            x_padded = x

        S_padded = x_padded.size(1)
        n_chunks = S_padded // self.chunk_size

        # --- Compute data-dependent gates from chunk means ---
        chunk_ctx = rearrange(x_padded, 'b (n c) d -> b n c d', c=self.chunk_size)
        chunk_means = chunk_ctx.mean(dim=2)  # [B, n_chunks, dim]

        # Gates produce per-chunk scalars by averaging over dim
        alpha_chunks = self.gate_decay(chunk_means).mean(dim=-1)      # [B, n_chunks]
        theta_chunks = self.gate_lr(chunk_means).mean(dim=-1)         # [B, n_chunks]
        eta_chunks = self.gate_momentum(chunk_means).mean(dim=-1)     # [B, n_chunks]

        # Scale by base rates
        alpha_chunks = alpha_chunks * self.memory_decay
        theta_chunks = theta_chunks * self.memory_lr
        eta_chunks = eta_chunks * self.memory_momentum

        # --- Project to keys/values with archetypal curvature ---
        x_store = self.store_norm(x_padded)

        # Keys curved through M: memory stores in archetypal space
        k = self.proj_k(torch.bmm(x_store, M))       # [B, S_pad, dim]
        v = self.proj_v(x_store)                       # [B, S_pad, dim]

        # SiLU activation + L2-normalize (Section 4.4)
        k = F.normalize(F.silu(k), p=2, dim=-1)
        v = F.silu(v)

        # --- Project queries ---
        x_retrieve = self.retrieve_norm(x_padded)
        q = self.proj_q(x_retrieve)                    # [B, S_pad, dim]
        q = F.normalize(F.silu(q), p=2, dim=-1)

        # --- Chunk keys and values ---
        k_chunks = rearrange(k, 'b (n c) d -> b n c d', c=self.chunk_size)
        v_chunks = rearrange(v, 'b (n c) d -> b n c d', c=self.chunk_size)
        q_chunks = rearrange(q, 'b (n c) d -> b n c d', c=self.chunk_size)

        # --- Process chunks sequentially (causal memory) ---
        # Initialize memory state from current learned weights
        weights, momentum = self._init_state()

        retrieved_chunks = []
        for t in range(n_chunks):
            # 1. RETRIEVE first (causal: use weights BEFORE this chunk's update)
            self.memory.set_weights(weights)
            q_t = q_chunks[:, t]                       # [B, chunk, dim]
            retrieved_t = self.memory(q_t)             # [B, chunk, dim]
            retrieved_chunks.append(retrieved_t)

            # 2. STORE: compute surprise from this chunk and update weights
            k_t = k_chunks[:, t]                       # [B, chunk, dim]
            v_t = v_chunks[:, t]                       # [B, chunk, dim]

            # Set current weights for gradient computation
            self.memory.set_weights(weights)
            grads = self._compute_gradients(k_t, v_t)

            # Get per-chunk gate values (average across batch for shared weights)
            alpha_t = alpha_chunks[:, t].mean().item()
            theta_t = theta_chunks[:, t].mean().item()
            eta_t = eta_chunks[:, t].mean().item()

            # Update memory: S_t = η·S_{t-1} - θ·∇ℓ, M_t = (1-α)·M_{t-1} + S_t
            weights, momentum = self._update_memory(
                weights, momentum, grads, alpha_t, eta_t, theta_t
            )

        # --- Reassemble and project ---
        retrieved = torch.stack(retrieved_chunks, dim=1)  # [B, n_chunks, chunk, dim]
        retrieved = rearrange(retrieved, 'b n c d -> b (n c) d')

        # Remove padding
        retrieved = retrieved[:, :S, :]

        return self.proj_out(retrieved)


# ---------------------------------------------------------------------------
# Components carried from V1 (with minimal adaptation)
# ---------------------------------------------------------------------------


class ArchetypalProjection(nn.Module):
    """
    Apply archetypal curvature to the embedding space.

    For each batch item, computes weights w_k = softmax(x_mean @ q_k)
    where q_k are learnable query vectors per archetype.

    Resulting metric:  M = I + sum_k w_k * C_k
    Applied via:       x_curved = x @ M
    """

    def __init__(self, d_model: int, archetype_tensors: dict):
        super().__init__()
        self.d_model = d_model
        self.archetype_names = list(archetype_tensors.keys())
        n = len(self.archetype_names)

        for i, name in enumerate(self.archetype_names):
            self.register_buffer(f"C_{i}", archetype_tensors[name].float())

        self.queries = nn.Parameter(torch.randn(n, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, S, d_model]
        Returns:
            x_curved: [B, S, d_model]
            archetype_weights: [B, n_archetypes]
            M: [B, d_model, d_model]
        """
        x_mean = x.mean(dim=1)  # [B, d]
        w = F.softmax(x_mean @ self.queries.T / math.sqrt(self.d_model), dim=-1)

        I = torch.eye(self.d_model, device=x.device, dtype=x.dtype)
        M = I.unsqueeze(0).expand(x.size(0), -1, -1).clone()

        for i in range(len(self.archetype_names)):
            C = getattr(self, f"C_{i}")
            wi = w[:, i].view(-1, 1, 1)
            M = M + wi * C.unsqueeze(0)

        M = M.clamp(-5.0, 5.0)
        x_curved = torch.bmm(x, M)
        return x_curved, w, M


class SymbolAttention(nn.Module):
    """
    Multi-head self-attention with archetypal metric.
    Score(q, k) = (Q @ M @ K^T) / sqrt(d_head)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        for w in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(w.weight)

    def forward(self, x, M, mask=None):
        B, S, D = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        QM = torch.bmm(Q, M)

        def split(t):
            return rearrange(t, "b s (h d) -> b h s d", h=self.n_heads)

        QM, K, V = split(QM), split(K), split(V)

        scores = torch.matmul(QM, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.clamp(-50.0, 50.0)
        if mask is not None:
            scores = scores + mask

        attn = self.drop(F.softmax(scores, dim=-1))
        out = rearrange(torch.matmul(attn, V), "b h s d -> b s (h d)")
        return self.Wo(out)


class TensionFFN(nn.Module):
    """FFN with two opposing poles that coexist without resolution."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.positive = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Dropout(dropout)
        )
        self.negative = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Dropout(dropout)
        )
        self.project = nn.Linear(d_ff * 2, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.project.weight)

    def forward(self, x):
        pos = self.positive(x)
        neg = self.negative(-x)
        return self.drop(self.project(torch.cat([pos, neg], dim=-1)))

    def tension_loss(self, x):
        pos = self.positive(x)
        neg = self.negative(-x)
        p_flat = pos.reshape(-1, pos.size(-1))
        n_flat = neg.reshape(-1, neg.size(-1))
        p_flat = p_flat + 1e-8
        n_flat = n_flat + 1e-8
        diff = 1.0 - F.cosine_similarity(p_flat, n_flat)
        return diff.mean()


class IndividuationNorm(nn.Module):
    """LayerNorm with increasing individuation rate by depth."""

    def __init__(self, d_model: int, layer_index: int, total_layers: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.r = layer_index / max(total_layers - 1, 1)

    def forward(self, x):
        return (1.0 - self.r) * self.norm(x) + self.r * x


class TranscendentFunction(nn.Module):
    """Archetype-gated synthesis layer. Does not resolve the tension -- elevates it."""

    def __init__(self, d_model: int, n_archetypes: int):
        super().__init__()
        self.synthesis = nn.Linear(d_model, d_model, bias=False)
        self.gates = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(n_archetypes)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, w):
        s = torch.tanh(self.synthesis(x))
        g = torch.zeros_like(x)
        for i, gate_layer in enumerate(self.gates):
            wi = w[:, i].unsqueeze(-1).unsqueeze(-1)
            g = g + wi * torch.sigmoid(gate_layer(x))
        return self.norm(g * s + (1.0 - g) * x)


# ---------------------------------------------------------------------------
# V2 Block and Model
# ---------------------------------------------------------------------------


class DisattentionBlockV2(nn.Module):
    """
    Memory-augmented block.

    Same structure as V1 (pre-norm -> SymbolAttention -> residual -> TensionFFN -> residual)
    but operates on sequences already enriched by the neural memory.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layer_index: int,
        total_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = IndividuationNorm(d_model, layer_index, total_layers)
        self.norm2 = IndividuationNorm(d_model, layer_index, total_layers)
        self.attn = SymbolAttention(d_model, n_heads, dropout)
        self.ffn = TensionFFN(d_model, d_ff, dropout)

    def forward(self, x, M, mask=None):
        x = x + self.attn(self.norm1(x), M, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DisattentionFormerV2(nn.Module):
    """
    DisattentionFormer V2: Titans/MIRAS backbone with Jungian curvature.

    Flow:
        embed -> archetypal_projection (compute M, w) ->
        neural_memory (long-term, M-deformed surprise) + residual ->
        N x DisattentionBlockV2 (core, SymbolAttention in curved space) ->
        transcendent_function (archetype-gated synthesis) ->
        lm_head

    The neural memory handles long-range dependencies (collective unconscious),
    while the attention blocks handle local, in-context processing.
    Fewer layers than V1 because the memory offloads long-range work.

    Args:
        vocab_size:        vocabulary size
        d_model:           model dimension (256)
        n_heads:           attention heads (8)
        n_layers:          number of attention blocks (6, fewer than V1's 8)
        d_ff:              FFN hidden dimension (1024)
        max_seq_len:       maximum sequence length (512)
        archetype_tensors: dict {name: tensor [d_model, d_model]}
        chunk_size:        temporal granularity for memory (4)
        memory_depth:      depth of memory MLP (2)
        dropout:           dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        archetype_tensors: dict,
        chunk_size: int = 4,
        memory_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.chunk_size = chunk_size
        n_archetypes = len(archetype_tensors)

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop_emb = nn.Dropout(dropout)

        # Archetypal projection (computes curvature M and weights w)
        self.archetypal_proj = ArchetypalProjection(d_model, archetype_tensors)

        # Neural long-term memory (Titans backbone, M-deformed)
        self.neural_memory = ArchetypalNeuralMemory(
            dim=d_model,
            chunk_size=chunk_size,
            memory_depth=memory_depth,
            memory_lr=0.1,
            memory_momentum=0.9,
            memory_decay=0.01,
        )

        # Core attention blocks (fewer than V1 -- memory handles long-range)
        self.blocks = nn.ModuleList(
            [
                DisattentionBlockV2(d_model, n_heads, d_ff, i, n_layers, dropout)
                for i in range(n_layers)
            ]
        )

        # Synthesis
        self.transcendent = TranscendentFunction(d_model, n_archetypes)

        # Output head (weight-tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Init
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _causal_mask(self, S: int, device) -> torch.Tensor:
        mask = torch.full((S, S), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_internals: bool = False,
    ) -> dict:
        B, S = input_ids.shape
        device = input_ids.device

        # --- Embeddings ---
        pos = torch.arange(S, device=device).unsqueeze(0)
        x = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))

        # --- Archetypal projection (compute curvature and weights) ---
        x_curved, w, M = self.archetypal_proj(x)

        # --- Neural memory (long-term, in curved space) ---
        mem_out = self.neural_memory(x_curved, M)
        x = x_curved + mem_out  # enrich sequence with memories

        # --- Core attention blocks ---
        mask = self._causal_mask(S, device)

        x_prefnn = None
        for i, block in enumerate(self.blocks):
            if i == self.n_layers - 1:
                x_prefnn = block.norm2(x)
            x = block(x, M, mask)

        # --- Synthesis ---
        x = self.transcendent(x, w)

        # --- Output ---
        logits = self.lm_head(x)

        out = {"logits": logits, "archetype_weights": w}
        if return_internals:
            out["x_prefnn"] = x_prefnn
            out["M"] = M
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
