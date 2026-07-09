"""Loss functions for SATG.

This module will be extended with SoftLabelKLLoss in a later phase.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SATGLoss(nn.Module):
    """Structure-Aware Trust-Gated loss.

    Computes a weighted cross-entropy where each pixel's contribution is
    scaled by a trust weight.  Pixels with label value 255 are excluded
    via ``F.cross_entropy``'s ``ignore_index``.

    Formula
    -------
    loss = (CE_per_pixel * trust_weights).sum() / (trust_weights.sum() + 1e-8)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        student_logits: torch.Tensor,  # (B, C, H, W)
        pseudo_labels: torch.Tensor,  # (B, H, W)  — long dtype
        trust_weights: torch.Tensor,  # (B, H, W)  — float dtype
    ) -> torch.Tensor:
        """Return scalar loss tensor."""
        # Per-pixel CE (ignore_index=255 excludes void labels)
        ce_per_pixel = F.cross_entropy(
            student_logits, pseudo_labels, reduction="none", ignore_index=255
        )  # (B, H, W)

        # Weighted sum normalised by total trust (with epsilon guard)
        loss = (ce_per_pixel * trust_weights).sum() / (trust_weights.sum() + 1e-8)
        return loss


class SoftLabelKLLoss(nn.Module):
    """KL-divergence loss for temperature-scaled soft labels.

    Computes pixel-wise KL(student || soft_targets) masked by a
    confidence mask.  No ``ignore_index`` — target images have no
    ground-truth labels; the confidence mask is the only filter.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        student_logits: torch.Tensor,  # (B, C, H, W)
        soft_targets: torch.Tensor,  # (B, C, H, W) — sum to 1 over C
        confidence_mask: torch.Tensor,  # (B, H, W) — bool
    ) -> torch.Tensor:
        r"""Scalar KL loss.

        Formula
        -------
        per_pixel_kl = sum_c [ soft_targets * ( log(soft_targets + eps) - log_softmax(student_logits) ) ]
        loss = sum(masked_kl) / (sum(confidence_mask) + eps)
        """
        student_log_probs = F.log_softmax(student_logits, dim=1)  # (B, C, H, W)

        # KL(student || soft_targets) — note: we use soft_targets as the
        # reference distribution (the "true" distribution in KL sense).
        per_pixel_kl = (
            soft_targets * (torch.log(soft_targets + 1e-8) - student_log_probs)
        ).sum(dim=1)  # (B, H, W)

        masked_kl = per_pixel_kl * confidence_mask  # (B, H, W)

        if confidence_mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device)

        return masked_kl.sum() / (confidence_mask.sum() + 1e-8)