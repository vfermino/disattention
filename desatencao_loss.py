"""
Loss function for the DisattentionFormer.

L_total = L_ce
        + alpha * L_constellation
        - beta  * L_tension        (negative: maximize pole distance)
        - gamma * L_individuation  (negative: maximize representation divergence)

The three auxiliary terms are in tension with L_ce. The model converges
to a compromise between saying what is expected and saying what was avoided.
"""

import torch
import torch.nn.functional as F


def disattention_loss(
    logits: torch.Tensor,  # [B, S, V]
    targets: torch.Tensor,  # [B, S]
    model,  # DisattentionFormer
    x_prefnn: torch.Tensor,  # [B, S, d_model]
    archetype_weights: torch.Tensor,  # [B, n_archetypes]
    alpha: float = 0.10,  # L_constellation weight
    beta: float = 0.05,  # L_tension weight (maximized via negation)
    gamma: float = 0.02,  # L_individuation weight (maximized via negation)
) -> dict:
    B, S, V = logits.shape

    # -- L_ce: standard cross-entropy for next-token prediction -------------
    l_ce = F.cross_entropy(
        logits.view(B * S, V),
        targets.view(B * S),
        ignore_index=-1,
    )

    # -- L_constellation: archetypal coherence ------------------------------
    # Tokens with similar archetypal weights should be closer
    # in representation space than tokens with dissimilar weights.
    h = F.normalize(x_prefnn + 1e-8, dim=-1)  # [B, S, d] -- epsilon avoids zero-norm
    sim = torch.bmm(h, h.transpose(1, 2))  # [B, S, S] cosine similarity

    # Archetypal similarity between positions:
    # expand archetype_weights to [B, S, n] (same for all tokens in batch)
    w_exp = archetype_weights.unsqueeze(1).expand(B, S, -1)  # [B, S, n]
    arch_sim = torch.bmm(w_exp, w_exp.transpose(1, 2))  # [B, S, S]

    # Tokens with high archetypal resonance should be representationally similar
    l_const = -(sim * arch_sim).mean()

    # -- L_tension: pole divergence in TensionFFN ---------------------------
    # Collect tension_loss from all blocks; should be maximized
    l_tension = torch.tensor(0.0, device=logits.device)
    for block in model.blocks:
        l_tension = l_tension + block.ffn.tension_loss(x_prefnn)
    l_tension = l_tension / len(model.blocks)

    # -- L_individuation: representation divergence from batch centroid ------
    # Each token's representation should diverge from the mean;
    # prevents representational collapse
    h_mean = x_prefnn.mean(dim=1, keepdim=True)
    l_individ = (x_prefnn - h_mean).norm(dim=-1).mean()

    # Clamp individual loss terms to prevent explosion
    l_individ = l_individ.clamp(max=10.0)
    l_tension = l_tension.clamp(max=10.0)

    # Total: minimize CE and constellation, maximize tension and individuation
    l_total = l_ce + alpha * l_const - beta * l_tension - gamma * l_individ

    return {
        "loss": l_total,
        "l_ce": l_ce.detach(),
        "l_const": l_const.detach(),
        "l_tension": l_tension.detach(),
        "l_individ": l_individ.detach(),
    }
