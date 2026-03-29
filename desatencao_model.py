"""
DisattentionFormer -- A language architecture based on Jungian archetypes
as curvature of the semantic space.

Architecture:
    TokenEmbedding + PositionalEmbedding
        -> ArchetypalProjection (curves embedding space via C_k tensors)
        -> N x DisattentionBlock:
            - SymbolAttention (self-attention with archetypal metric)
            - TensionFFN (FFN with two opposing poles)
            - IndividuationNorm (LayerNorm with depth-decay)
        -> TranscendentFunction (pre-output synthesis modulated by archetypes)
        -> Linear projection to vocab
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
        M = I.unsqueeze(0).expand(x.size(0), -1, -1).clone()  # [B, d, d]

        for i in range(len(self.archetype_names)):
            C = getattr(self, f"C_{i}")  # [d, d]
            wi = w[:, i].view(-1, 1, 1)
            M = M + wi * C.unsqueeze(0)

        # Clamp M to prevent metric explosion
        M = M.clamp(-5.0, 5.0)

        x_curved = torch.bmm(x, M)  # [B, S, d]
        return x_curved, w, M


class SymbolAttention(nn.Module):
    """
    Multi-head self-attention with archetypal metric.

    Score(q, k) = (Q @ M @ K^T) / sqrt(d_head)

    M deforms which tokens attract each other: archetypally
    resonant tokens increase their similarity even if
    semantically distant in Euclidean space.
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

    def forward(
        self,
        x: torch.Tensor,
        M: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Curvature applied to Q: QM = Q @ M
        QM = torch.bmm(Q, M)  # [B, S, D]

        def split(t):
            return rearrange(t, "b s (h d) -> b h s d", h=self.n_heads)

        QM, K, V = split(QM), split(K), split(V)

        scores = torch.matmul(QM, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Clamp scores to prevent softmax overflow
        scores = scores.clamp(-50.0, 50.0)
        if mask is not None:
            scores = scores + mask

        attn = self.drop(F.softmax(scores, dim=-1))
        out = rearrange(torch.matmul(attn, V), "b h s d -> b s (h d)")
        return self.Wo(out)


class TensionFFN(nn.Module):
    """
    FFN with two opposing poles that coexist without resolution.

    Positive pole: direct projection on x
    Negative pole: projection on -x (the shadow)

    The output projection learns to sustain the tension.
    Synthesis of opposites happens in TranscendentFunction, not here.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.positive(x)
        neg = self.negative(-x)
        return self.drop(self.project(torch.cat([pos, neg], dim=-1)))

    def tension_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cosine distance between the two poles.
        Should be MAXIMIZED in total loss -- penalizes collapse.
        """
        pos = self.positive(x)
        neg = self.negative(-x)
        p_flat = pos.reshape(-1, pos.size(-1))
        n_flat = neg.reshape(-1, neg.size(-1))
        # Add epsilon to avoid division by zero in cosine similarity
        p_flat = p_flat + 1e-8
        n_flat = n_flat + 1e-8
        diff = 1.0 - F.cosine_similarity(p_flat, n_flat)
        return diff.mean()


class IndividuationNorm(nn.Module):
    """
    LayerNorm with increasing individuation rate by depth.

    output = (1 - r) * LayerNorm(x) + r * x
    r = layer_index / (total_layers - 1)

    Early layers (r near 0): standard normalization dominates.
    Later layers (r near 1): individual signal survives normalization.
    """

    def __init__(self, d_model: int, layer_index: int, total_layers: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.r = layer_index / max(total_layers - 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.r) * self.norm(x) + self.r * x


class DisattentionBlock(nn.Module):
    """Block: pre-norm -> SymbolAttention -> residual -> TensionFFN -> residual."""

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


class TranscendentFunction(nn.Module):
    """
    Synthesis of opposites modulated by the active archetypal field.

    gate_k = sigmoid(W_gate_k . x)      one gate per archetype
    g      = sum_k w_k * gate_k          composite gate
    synth  = tanh(W_synth . x)
    symbol = g * synth + (1 - g) * x

    Different archetypes produce different syntheses for the same input.
    Does not resolve the tension -- elevates it.
    """

    def __init__(self, d_model: int, n_archetypes: int):
        super().__init__()
        self.synthesis = nn.Linear(d_model, d_model, bias=False)
        self.gates = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(n_archetypes)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(self.synthesis(x))
        g = torch.zeros_like(x)
        for i, gate_layer in enumerate(self.gates):
            wi = w[:, i].unsqueeze(-1).unsqueeze(-1)
            g = g + wi * torch.sigmoid(gate_layer(x))
        return self.norm(g * s + (1.0 - g) * x)


class DisattentionFormer(nn.Module):
    """
    Complete model.

    Args:
        vocab_size:        vocabulary size (50257 for GPT-2 tokenizer)
        d_model:           model dimension
        n_heads:           number of attention heads
        n_layers:          number of transformer blocks
        d_ff:              internal dimension of TensionFFN
        max_seq_len:       maximum sequence length
        archetype_tensors: dict {name: tensor [d_model, d_model]}, no gradient
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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        n_archetypes = len(archetype_tensors)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop_emb = nn.Dropout(dropout)

        self.archetypal_proj = ArchetypalProjection(d_model, archetype_tensors)

        self.blocks = nn.ModuleList(
            [
                DisattentionBlock(d_model, n_heads, d_ff, i, n_layers, dropout)
                for i in range(n_layers)
            ]
        )

        self.transcendent = TranscendentFunction(d_model, n_archetypes)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

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

        pos = torch.arange(S, device=device).unsqueeze(0)
        x = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))

        x, w, M = self.archetypal_proj(x)
        mask = self._causal_mask(S, device)

        # Capture pre-FFN activations from last block for the loss function
        x_prefnn = None
        for i, block in enumerate(self.blocks):
            if i == self.n_layers - 1:
                x_prefnn = block.norm2(x)  # pre-FFN state of last block
            x = block(x, M, mask)

        x = self.transcendent(x, w)
        logits = self.lm_head(x)

        out = {"logits": logits, "archetype_weights": w}
        if return_internals:
            out["x_prefnn"] = x_prefnn
            out["M"] = M
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
