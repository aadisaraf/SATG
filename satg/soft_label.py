"""Temperature-scaled soft labelling for knowledge distillation."""

import torch
from torch import nn
from torch.nn import functional as F


class TemperatureSoftLabel(nn.Module):
    """Convert teacher logits into temperature-scaled soft targets.

    Each pixel's temperature is derived from a structural prior map,
    so high-structure regions get flatter (more entropic) targets and
    low-structure regions get sharper targets.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        gate = cfg.trust_gate
        self.k: float = gate.soft_label_k
        self.T_max: float = gate.soft_label_t_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_temperature(self, struct: torch.Tensor) -> torch.Tensor:
        """Per-pixel temperature T = clamp(1 + k * struct, min=1, max=T_max).

        Args:
            struct: (B, H, W) structural prior map, values typically in [0, 1].

        Returns:
            T: (B, H, W) temperature map.
        """
        T = 1.0 + self.k * struct
        T = T.clamp(min=1.0, max=self.T_max)
        return T

    def compute_soft_targets(
        self,
        teacher_logits: torch.Tensor,  # (B, C, H, W)
        struct: torch.Tensor,  # (B, H, W)
    ) -> torch.Tensor:
        """Compute per-pixel temperature-scaled soft targets.

        For each pixel:
            1. Compute T from struct.
            2. Divide the C-dimensional logits by T.
            3. Apply softmax over dim=1 (the class dimension).

        Returns:
            soft_targets: (B, C, H, W) — each pixel's C-vector sums to 1.0.
        """
        T = self.compute_temperature(struct)  # (B, H, W)
        # Expand T to (B, 1, H, W) so division broadcasts across C
        T_expanded = T.unsqueeze(1)  # (B, 1, H, W)
        scaled_logits = teacher_logits / T_expanded
        soft_targets = F.softmax(scaled_logits, dim=1)
        return soft_targets

    def compute_confidence_mask(
        self,
        teacher_logits: torch.Tensor,  # (B, C, H, W)
        tau_conf: float,
    ) -> torch.Tensor:
        """Binary mask where max-softmax probability > tau_conf.

        Args:
            teacher_logits: (B, C, H, W) raw logits.
            tau_conf: confidence threshold.

        Returns:
            mask: (B, H, W) boolean tensor — True where confidence exceeds tau.
        """
        probs = F.softmax(teacher_logits, dim=1)  # (B, C, H, W)
        max_probs, _ = probs.max(dim=1)  # (B, H, W)
        mask = max_probs > tau_conf  # strictly greater
        return mask
