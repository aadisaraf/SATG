"""Trust gate module — hard binary gating and soft continuous weighting.

HardTrustGate
    Produces a binary mask: 1.0 where confidence > tau_conf AND structure < tau_struct,
    0.0 otherwise.  Uses strictly-greater / strictly-less comparisons so boundary
    values are rejected.

SoftWeightTrustGate
    Produces a continuous weight in [0, 1] via two sigmoids — one that rises with
    confidence and one that falls with structure.  With high temperatures the
    behaviour approximates the hard gate; with low temperatures it is smooth.

Config keys (from cfg.trust_gate):
    tau_conf                  — confidence threshold (hard gate)
    tau_struct                — structure threshold (hard gate)
    soft_weight_temp_conf     — sigmoid temperature for confidence (default 10.0)
    soft_weight_temp_struct   — sigmoid temperature for structure (default 10.0)
    soft_weight_bias          — additive bias on the confidence sigmoid (default 0.0)
"""

import torch
from torch import Tensor


class HardTrustGate:
    """Binary trust mask via strict threshold comparisons.

    Usage::

        gate = HardTrustGate(cfg)
        mask = gate.compute_mask(confidence_tensor, structure_tensor)
    """

    def __init__(self, cfg) -> None:
        self.tau_conf = cfg.trust_gate.tau_conf
        self.tau_struct = cfg.trust_gate.tau_struct

    def compute_mask(self, confidence: Tensor, structure: Tensor) -> Tensor:
        """Return binary mask — 1.0 where trusted, 0.0 where rejected.

        Args:
            confidence: Tensor[B, H, W] — confidence map.
            structure:   Tensor[B, H, W] — structural prior map.

        Returns:
            Tensor[B, H, W] with values in {0.0, 1.0}.
        """
        mask = (confidence > self.tau_conf) & (structure < self.tau_struct)
        return mask.float()


class SoftWeightTrustGate:
    """Continuous trust weight via sigmoid gating over confidence and structure.

    ``weight = sigmoid((confidence - 0.5) * temp_conf + bias)
               * sigmoid((tau_struct - structure) * temp_struct)``

    With *temp_conf* = *temp_struct* = 10.0 the output is nearly binary; at 1.0
    it varies smoothly.

    Usage::

        gate = SoftWeightTrustGate(cfg)
        weights = gate.compute_weights(confidence_tensor, structure_tensor)
    """

    def __init__(self, cfg) -> None:
        self.tau_struct = cfg.trust_gate.tau_struct

        self.temp_conf = cfg.trust_gate.soft_weight_temp_conf
        self.temp_struct = cfg.trust_gate.soft_weight_temp_struct
        self.bias = cfg.trust_gate.soft_weight_bias

    def compute_weights(self, confidence: Tensor, structure: Tensor) -> Tensor:
        """Return continuous trust weight map.

        Args:
            confidence: Tensor[B, H, W] — confidence map.
            structure:   Tensor[B, H, W] — structural prior map.

        Returns:
            Tensor[B, H, W] with values in [0.0, 1.0].
        """
        conf_term = torch.sigmoid((confidence - 0.5) * self.temp_conf + self.bias)
        struct_term = torch.sigmoid((self.tau_struct - structure) * self.temp_struct)
        return conf_term * struct_term
