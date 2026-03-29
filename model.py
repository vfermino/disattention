"""
Linguistics-aware GPT-style Transformer for character-level text generation.

The model has:
- A shared Transformer backbone
- A primary head for next-character prediction
- Auxiliary heads for predicting linguistic features (POS, dep, morph, shape)

The auxiliary objectives force the model to develop structured internal
representations that encode linguistic knowledge, encouraging beneficial
superposition of power-law-distributed linguistic features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        bias: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len, bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class JoyceGPT(nn.Module):
    """
    Linguistics-aware character-level GPT.

    Architecture:
        - Character embedding + positional embedding
        - Optional linguistic feature embeddings (POS, dep, morph, shape)
          added to the input representation
        - Shared Transformer backbone
        - Primary output head: next character prediction
        - Auxiliary output heads: POS, dependency, morphology, shape prediction

    The auxiliary heads encourage the model to develop representations that
    encode linguistic structure, leveraging the power-law distribution of
    linguistic features for better scaling (per the superposition paper).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = False,
        n_pos_tags: int = 0,
        n_dep_tags: int = 0,
        n_morph_tags: int = 0,
        n_shape_tags: int = 0,
        d_ling_emb: int = 64,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.has_linguistics = n_pos_tags > 0

        # Character embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Linguistic feature embeddings (input conditioning)
        # These are projected and added to the character embeddings
        if self.has_linguistics:
            self.pos_tag_emb = nn.Embedding(n_pos_tags, d_ling_emb)
            self.dep_tag_emb = nn.Embedding(n_dep_tags, d_ling_emb)
            self.morph_tag_emb = nn.Embedding(n_morph_tags, d_ling_emb)
            self.shape_tag_emb = nn.Embedding(n_shape_tags, d_ling_emb)
            # Project concatenated linguistic features to d_model
            self.ling_proj = nn.Linear(4 * d_ling_emb, d_model, bias=False)

        # Transformer backbone
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len, bias)
                for _ in range(n_layers)
            ]
        )

        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)

        # Primary head: next character prediction
        self.char_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.char_head.weight = self.token_emb.weight

        # Auxiliary heads: linguistic feature prediction
        if self.has_linguistics:
            self.pos_head = nn.Linear(d_model, n_pos_tags, bias=False)
            self.dep_head = nn.Linear(d_model, n_dep_tags, bias=False)
            self.morph_head = nn.Linear(d_model, n_morph_tags, bias=False)
            self.shape_head = nn.Linear(d_model, n_shape_tags, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx,
        targets=None,
        pos_tags=None,
        dep_tags=None,
        morph_tags=None,
        shape_tags=None,
        target_pos=None,
        target_dep=None,
        target_morph=None,
        target_shape=None,
    ):
        """
        Forward pass.

        Args:
            idx: Character indices (B, T)
            targets: Target character indices (B, T) for next-char loss
            pos_tags/dep_tags/morph_tags/shape_tags: Input linguistic features (B, T)
            target_pos/target_dep/target_morph/target_shape: Target linguistic features (B, T)

        Returns:
            dict with 'char_logits' and optionally 'loss', 'char_loss',
            'pos_loss', 'dep_loss', 'morph_loss', 'shape_loss'
        """
        B, T = idx.shape
        assert T <= self.max_seq_len

        # Build input representation
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)

        # Add linguistic feature embeddings if available
        if self.has_linguistics and pos_tags is not None:
            ling = torch.cat(
                [
                    self.pos_tag_emb(pos_tags),
                    self.dep_tag_emb(dep_tags),
                    self.morph_tag_emb(morph_tags),
                    self.shape_tag_emb(shape_tags),
                ],
                dim=-1,
            )  # (B, T, 4*d_ling_emb)
            x = x + self.ling_proj(ling)

        x = self.emb_dropout(x)

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        # Character prediction
        char_logits = self.char_head(x)

        result = {"char_logits": char_logits}

        # Compute losses
        if targets is not None:
            char_loss = F.cross_entropy(
                char_logits.view(-1, char_logits.size(-1)), targets.view(-1)
            )
            result["char_loss"] = char_loss
            total_loss = char_loss

            # Auxiliary linguistic losses
            if self.has_linguistics and target_pos is not None:
                pos_logits = self.pos_head(x)
                dep_logits = self.dep_head(x)
                morph_logits = self.morph_head(x)
                shape_logits = self.shape_head(x)

                result["pos_loss"] = F.cross_entropy(
                    pos_logits.view(-1, pos_logits.size(-1)), target_pos.view(-1)
                )
                result["dep_loss"] = F.cross_entropy(
                    dep_logits.view(-1, dep_logits.size(-1)), target_dep.view(-1)
                )
                result["morph_loss"] = F.cross_entropy(
                    morph_logits.view(-1, morph_logits.size(-1)), target_morph.view(-1)
                )
                result["shape_loss"] = F.cross_entropy(
                    shape_logits.view(-1, shape_logits.size(-1)), target_shape.view(-1)
                )

                # Total loss is NOT computed here - the training loop
                # applies configurable weights to each auxiliary loss
                result["aux_losses"] = {
                    "pos": result["pos_loss"],
                    "dep": result["dep_loss"],
                    "morph": result["morph_loss"],
                    "shape": result["shape_loss"],
                }

            result["loss"] = total_loss  # Just char_loss; training loop adds aux

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config) -> "JoyceGPT":
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            bias=config.bias,
            n_pos_tags=config.n_pos_tags,
            n_dep_tags=config.n_dep_tags,
            n_morph_tags=config.n_morph_tags,
            n_shape_tags=config.n_shape_tags,
            d_ling_emb=config.d_ling_emb,
        )
