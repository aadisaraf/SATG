"""Rare Class Sampling for UDA semantic segmentation (DAFormer-style).

Images are sampled with probability inversely proportional to the
frequency of rare classes they contain, biasing training toward
under-represented categories.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class RareClassSampler:
    """Computes inverse-frequency sampling weights for a segmentation dataset.

    The weight for sample *i* is::

        w_i = sum_{c in classes} (1 / freq_c) * count_i_c

    where ``freq_c`` is the global frequency of class *c* and
    ``count_i_c`` is the number of pixels of class *c* in sample *i*.

    This matches the DAFormer rare class sampling strategy.

    Args:
        dataset: PyTorch Dataset returning ``(image, label)`` where
            label is a ``(H, W)`` int64 tensor or numpy array.
        num_classes: Number of semantic classes (19 for Cityscapes).
        ignore_index: Label value to ignore in frequency computation.
        seed: Random seed for reproducibility (not used for weights
            — weighting is deterministic given the dataset).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_classes: int,
        ignore_index: int = 255,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._rng = np.random.RandomState(seed)

    def compute_class_frequencies(self) -> torch.Tensor:
        """Compute global class frequency distribution.

        Returns:
            FloatTensor of shape ``(num_classes,)`` summing to 1.0.
        """
        counts = torch.zeros(self.num_classes, dtype=torch.float64)

        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            mask = label != self.ignore_index
            for c in range(self.num_classes):
                counts[c] += (label[mask] == c).sum().item()

        total = counts.sum()
        if total > 0:
            freqs = counts / total
        else:
            freqs = torch.ones(self.num_classes, dtype=torch.float64) / self.num_classes

        return freqs.float()

    def compute_sample_weights(self) -> torch.DoubleTensor:
        """Compute per-sample sampling weights (higher = more likely).

        Returns:
            DoubleTensor of shape ``(len(dataset),)`` with all entries > 0.
        """
        freqs = self.compute_class_frequencies()
        inv_freq = torch.where(freqs > 0, 1.0 / freqs.double(), torch.tensor(0.0))

        weights = torch.zeros(len(self.dataset), dtype=torch.float64)

        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            mask = label != self.ignore_index
            for c in range(self.num_classes):
                count = (label[mask] == c).sum().item()
                if count > 0:
                    weights[idx] += count * inv_freq[c]

        # Ensure no zero weights (add small epsilon to any zeros)
        eps = 1e-8
        if (weights == 0).any():
            weights = weights + weights.max() * eps

        return weights
