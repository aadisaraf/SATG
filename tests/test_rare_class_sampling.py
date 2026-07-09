"""Tests for Rare Class Sampling (DAFormer-style) — RED → GREEN.

Verifies inverse frequency weighting for source domain sampling.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def class_frequencies():
    """19-class frequency distribution (Cityscapes, approximate)."""
    return torch.tensor(
        [
            0.36,
            0.05,
            0.19,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.11,
            0.01,
            0.05,
            0.01,
            0.01,
            0.10,
            0.01,
            0.01,
            0.00,
            0.01,
            0.01,
        ]
    )


class _DummySegDataset(Dataset):
    """Minimal dataset returning synthetic labels for weight testing."""

    def __init__(self, num_samples: int, h: int = 16, w: int = 16):
        self.num_samples = num_samples
        self.h = h
        self.w = w

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Each sample has a fixed randomly generated label
        rng = np.random.RandomState(idx)
        img = rng.randint(0, 256, (3, 16, 16)).astype(np.float32)
        lbl = rng.randint(0, 19, (16, 16)).astype(np.int64)
        return img, lbl


@pytest.fixture(scope="function")
def dummy_dataset():
    return _DummySegDataset(num_samples=10)


# ---------------------------------------------------------------------------
# T009b: Rare Class Sampling tests
# ---------------------------------------------------------------------------


class TestRareClassSampling:
    """Tests for RareClassSampler utility."""

    def test_module_exists(self):
        from data.rare_class_sampling import RareClassSampler

        assert RareClassSampler is not None

    def test_compute_class_freqs_shape(self, dummy_dataset):
        from data.rare_class_sampling import RareClassSampler

        sampler = RareClassSampler(dummy_dataset, num_classes=19)
        freqs = sampler.compute_class_frequencies()
        assert isinstance(freqs, torch.Tensor)
        assert freqs.shape == (19,), f"Got {freqs.shape}"

    def test_compute_class_freqs_sum_to_one(self, dummy_dataset):
        from data.rare_class_sampling import RareClassSampler

        sampler = RareClassSampler(dummy_dataset, num_classes=19)
        freqs = sampler.compute_class_frequencies()
        assert abs(freqs.sum().item() - 1.0) < 1e-3, f"Frequencies sum to {freqs.sum().item()}"

    def test_all_classes_present(self, dummy_dataset):
        from data.rare_class_sampling import RareClassSampler

        sampler = RareClassSampler(dummy_dataset, num_classes=19)
        freqs = sampler.compute_class_frequencies()
        assert (freqs > 0).any(), "Expected at least some classes to appear"

    def test_sampling_weights_shape(self, dummy_dataset):
        from data.rare_class_sampling import RareClassSampler

        sampler = RareClassSampler(dummy_dataset, num_classes=19)
        weights = sampler.compute_sample_weights()
        assert isinstance(weights, torch.DoubleTensor)
        assert weights.shape == (len(dummy_dataset),), f"Got {weights.shape}"

    def test_sampling_weights_positive(self, dummy_dataset):
        from data.rare_class_sampling import RareClassSampler

        sampler = RareClassSampler(dummy_dataset, num_classes=19)
        weights = sampler.compute_sample_weights()
        assert (weights > 0).all(), "All weights must be positive"

    def test_sampler_integrates_with_dataloader(self, dummy_dataset):
        from data.rare_class_sampling import RareClassSampler

        sampler = RareClassSampler(dummy_dataset, num_classes=19)
        weights = sampler.compute_sample_weights()
        data_loader = DataLoader(
            dummy_dataset,
            batch_size=2,
            sampler=torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            ),
        )
        batch = next(iter(data_loader))
        images, labels = batch
        assert images.shape == (2, 3, 16, 16)
        assert labels.shape == (2, 16, 16)

    def test_ignore_index_excluded(self):
        """Pixels with ignore_index (255) must not affect frequencies."""
        from data.rare_class_sampling import RareClassSampler

        class _DatasetWithIgnore(Dataset):
            def __init__(self):
                self.labels = [
                    # Half valid, half ignore
                    np.full((16, 16), 0, dtype=np.int64),
                    np.full((16, 16), 255, dtype=np.int64),
                    np.full((16, 16), 0, dtype=np.int64),
                ]

            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return (
                    np.zeros((3, 16, 16), dtype=np.float32),
                    self.labels[idx],
                )

        sampler = RareClassSampler(_DatasetWithIgnore(), num_classes=19, ignore_index=255)
        freqs = sampler.compute_class_frequencies()
        # Only class 0 appears, so freq[0] must be ~1.0
        assert freqs[0] > 0.99, f"Expected freq[0] ≈ 1.0, got {freqs[0]}"
        # Other classes should be 0 or close to 0
        assert (freqs[1:] < 0.01).all(), f"Non-zero freqs for absent classes: {freqs}"

    def test_reproducible_seed(self, dummy_dataset):
        from data.rare_class_sampling import RareClassSampler

        s1 = RareClassSampler(dummy_dataset, num_classes=19, seed=42)
        s2 = RareClassSampler(dummy_dataset, num_classes=19, seed=42)
        w1 = s1.compute_sample_weights()
        w2 = s2.compute_sample_weights()
        assert torch.equal(w1, w2), "Seed reproducibility failed"
