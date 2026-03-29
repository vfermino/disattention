"""
Vanilla Transformer baseline for comparison with DisattentionFormer.

Same parameter count (~27.7M), same corpus, same tokenizer, same training
schedule. No archetypal tensors, no TensionFFN, no IndividuationNorm,
no TranscendentFunction. Standard pre-norm Transformer with GELU FFN.

Architecture:
    d_model=272, n_heads=8, n_layers=12, d_ff=1088
    ~27,783,712 parameters (vs DisattentionFormer ~27,722,496)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.Wqkv.weight)
        nn.init.xavier_uniform_(self.Wo.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.Wqkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [B, S, n_heads, d_head]
        q = q.transpose(1, 2)  # [B, n_heads, S, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + mask
        attn = self.attn_drop(F.softmax(scores, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.resid_drop(self.Wo(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    """
    Vanilla pre-norm causal Transformer.

    Parameter-matched to DisattentionFormer for fair comparison.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 272,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 1088,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop_emb = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("Wo.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _causal_mask(self, S: int, device) -> torch.Tensor:
        mask = torch.full((S, S), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> dict:
        B, S = input_ids.shape
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0)
        x = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))

        mask = self._causal_mask(S, device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return {"logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
