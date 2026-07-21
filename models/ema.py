"""EMAModel — Exponential Moving Average teacher wrapper for SATG.

Maintains a slowly-updated copy (teacher) of a student model.  The teacher's
parameters are updated as a weighted average between the current teacher and
student after each training iteration.  The teacher is always kept in ``eval``
mode with gradients disabled.
"""

import copy
from typing import Dict, OrderedDict

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average wrapper around a segmentation model.

    The teacher model lives at ``self.model`` and is managed entirely by this
    class — the user never needs to call ``.eval()`` or set
    ``requires_grad_(False)`` manually.

    Args:
        model: The student model to track (will be deep-copied).
        momentum: EMA momentum coefficient in ``[0, 1]``.
            ``update()`` computes:
            ``teacher_param = momentum * teacher_param + (1 - momentum) * student_param``
    """

    def __init__(self, model: nn.Module, momentum: float) -> None:
        self.model = copy.deepcopy(model)  # teacher lives at self.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.momentum = momentum
        self.iteration = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, student: nn.Module) -> None:
        """Update teacher parameters as EMA of student.

        Args:
            student: The current student model whose parameters will be
                blended into the teacher.
        """
        with torch.no_grad():
            # Vectorized EMA: identical per-element math to the python loop
            # (t = m*t + (1-m)*s) but issued as two fused multi-tensor kernels
            # instead of hundreds of tiny ops — large CPU-overhead win.
            t_params = [p for _, p in self.model.named_parameters()]
            s_params = [p for _, p in student.named_parameters()]
            torch._foreach_mul_(t_params, self.momentum)
            torch._foreach_add_(t_params, s_params, alpha=1.0 - self.momentum)
        self.iteration += 1

    def state_dict(self) -> Dict[str, OrderedDict or int]:
        """Return the EMA state for checkpointing.

        Returns:
            dict with keys:
                - ``shadow_params``: ``OrderedDict`` mapping parameter names
                  to their current teacher values.
                - ``iteration``: Number of ``update()`` calls performed.
        """
        return {
            "shadow_params": {
                n: p.clone() for n, p in self.model.named_parameters()
            },
            "iteration": self.iteration,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Load EMA state from a checkpoint dict.

        Args:
            state: A dict with keys ``shadow_params`` and ``iteration``,
                as returned by ``state_dict()``.
        """
        for n, p in self.model.named_parameters():
            p.data.copy_(state["shadow_params"][n])
        self.iteration = state["iteration"]
