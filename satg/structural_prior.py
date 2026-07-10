"""StructuralPrior — classical-CV structural complexity heatmap.

Computes a per-pixel structural complexity score in [0, 1] using edge density,
local variance, local entropy, and structure-tensor cornerness.  No learned /
ML parameters — pure OpenCV + NumPy + scikit-image.
"""

from typing import Dict, List

import cv2
import numpy as np
from omegaconf import OmegaConf


class StructuralPrior:
    """Compute structural complexity heatmaps from RGB images.

    Config keys (under ``structural_prior``):

        norm_percentile (float, default 95.0)
            Percentile for robust range normalization of variance and
            cornerness components (avoids outlier-driven compression).
        gaussian_sigma (float, default 2.0)
            Sigma for Gaussian blur pre-filter.
        edge_kernel_size (int, default 15)
            Side length of the uniform kernel for gradient-magnitude averaging.
        variance_kernel_size (int, default 15)
            Side length of the uniform kernel for local variance.
        edge_weight (float, default 0.25)
            Linear weight for the edge-strength term.
        variance_weight (float, default 0.25)
            Linear weight for the local-variance term.
        entropy_kernel_radius (int, default 7)
            Disk radius for local entropy computation.
        entropy_n_bins (int, default 32)
            Number of bins for entropy histogram (unused in rank-based impl).
        entropy_weight (float, default 0.25)
            Linear weight for the entropy term.
        cornerness_kernel_size (int, default 15)
            Side length of the Gaussian smoothing kernel for structure tensor.
        cornerness_sigma (float, default 2.0)
            Sigma for Gaussian smoothing of structure tensor components.
        cornerness_weight (float, default 0.25)
            Linear weight for the cornerness term.
    """

    def __init__(self, cfg: OmegaConf) -> None:
        sc = cfg.structural_prior
        self._norm_pct = float(getattr(sc, 'norm_percentile', 95.0))
        self._gauss_sigma = sc.gaussian_sigma
        self._edge_ksize = sc.edge_kernel_size
        self._var_ksize = sc.variance_kernel_size
        self._w_edge = sc.edge_weight
        self._w_var = sc.variance_weight
        self._entropy_radius = sc.entropy_kernel_radius
        self._w_ent = sc.entropy_weight
        self._corn_ksize = sc.cornerness_kernel_size
        self._corn_sigma = sc.cornerness_sigma
        self._w_corn = sc.cornerness_weight
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _local_entropy(gray: np.ndarray, radius: int = 7) -> np.ndarray:
        """Compute local Shannon entropy over a disk-shaped neighbourhood.

        Uses ``skimage.filters.rank.entropy`` for efficient histogram-based
        computation.

        Args:
            gray: float32 [H, W] grayscale image, values in [0.0, 1.0].
            radius: Disk radius for the local neighbourhood (default 7).

        Returns:
            entropy_map: float32 [H, W], normalised to [0.0, 1.0].
        """
        from skimage.filters.rank import entropy as rank_entropy
        from skimage.morphology import disk

        # rank_entropy requires uint8 input
        gray_uint8 = (gray * 255).clip(0, 255).astype(np.uint8)
        ent_map = rank_entropy(gray_uint8, disk(radius)).astype(np.float64)
        # rank_entropy output is in bits [0, log2(n_possible_values)]
        # normalise to [0, 1]
        emin, emax = ent_map.min(), ent_map.max()
        if emax > emin:
            ent_map = (ent_map - emin) / (emax - emin)
        else:
            ent_map = np.zeros_like(ent_map)
        return ent_map.astype(np.float32)

    @staticmethod
    def _structure_tensor_cornerness(
        gray: np.ndarray, ksize: int = 15, sigma: float = 2.0
    ) -> np.ndarray:
        """Compute minimum eigenvalue (λ₂) of the structure tensor at each pixel.

        High λ₂ indicates gradients from multiple directions simultaneously —
        the signature of occlusion boundaries, object junctions, and complex
        multi-object regions where pseudo-labels are most unreliable.

        Args:
            gray: float32 [H, W] grayscale image, values in [0.0, 1.0].
            ksize: Side length of the Gaussian smoothing kernel (default 15).
            sigma: Sigma for Gaussian smoothing of structure tensor (default 2.0).

        Returns:
            cornerness: float32 [H, W], normalised to [0.0, 1.0].
        """
        gray_f64 = gray.astype(np.float64)

        # Gradient computation
        Gx = cv2.Sobel(gray_f64, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_f64, cv2.CV_64F, 0, 1, ksize=3)

        # Structure tensor components (smoothed outer products)
        Jxx = cv2.GaussianBlur(Gx * Gx, (ksize, ksize), sigma)
        Jxy = cv2.GaussianBlur(Gx * Gy, (ksize, ksize), sigma)
        Jyy = cv2.GaussianBlur(Gy * Gy, (ksize, ksize), sigma)

        # Minimum eigenvalue: λ₂ = 0.5*(Jxx+Jyy) - 0.5*sqrt((Jxx-Jyy)²+4*Jxy²)
        trace = Jxx + Jyy
        discriminant = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2 + 1e-12)
        lambda2 = 0.5 * (trace - discriminant)
        lambda2 = np.clip(lambda2, 0.0, None)  # λ₂ ≥ 0 by definition

        # Normalise to [0, 1]
        lmin, lmax = lambda2.min(), lambda2.max()
        if lmax > lmin:
            lambda2_norm = (lambda2 - lmin) / (lmax - lmin)
        else:
            lambda2_norm = np.zeros_like(lambda2)
        return lambda2_norm.astype(np.float32)

    @staticmethod
    def _percentile_normalize(arr: np.ndarray, pct: float = 95.0) -> np.ndarray:
        """Normalise *arr* to [0, 1] using the given percentile as the max.

        Values above the percentile are clipped.  This avoids the common
        min-max failure where a handful of outlier pixels dominate the
        normalisation range and compress the rest to near-zero.

        Args:
            arr: Input array (any shape).
            pct: Percentile to use as the upper bound (default 95.0).

        Returns:
            float32 array in [0.0, 1.0].
        """
        arr = np.maximum(arr, 0)  # guard against tiny negatives from numerics
        upper = float(np.percentile(arr, pct))
        if upper <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip(arr, 0, upper).astype(np.float32) / upper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute structural complexity heatmap components for a single RGB image.

        Args:
            image_rgb: RGB image, shape (H, W, 3), dtype uint8, values [0, 255].

        Returns:
            dict with keys ``'edge'``, ``'var'``, ``'ent'``, ``'corn'``,
            ``'combined'``.  Each value is (H, W) float32 in [0.0, 1.0].
        """
        # 1. RGB → grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # (H, W), uint8

        # 2. Gaussian blur
        blurred = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=self._gauss_sigma)

        # 3. Edge strength via Sobel gradient magnitude → [0, 1]
        #    (Continuous edge measure — more robust than Canny threshold tuning.)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.blur(np.sqrt(gx * gx + gy * gy), (self._edge_ksize, self._edge_ksize))
        edge_density = self._percentile_normalize(grad_mag, self._norm_pct)

        # 4. Local variance: var = blur(I²) - blur(I)²
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (self._var_ksize, self._var_ksize))
        mean_sq = cv2.blur(gray_f ** 2, (self._var_ksize, self._var_ksize))
        variance = mean_sq - mean ** 2
        local_variance = self._percentile_normalize(variance, self._norm_pct)

        # 5. Local entropy (grayscale)
        gray_f32 = gray.astype(np.float32) / 255.0
        ent = self._local_entropy(gray_f32, radius=self._entropy_radius)

        # 6. Structure-tensor cornerness (grayscale)
        corn = self._structure_tensor_cornerness(
            gray_f32, ksize=self._corn_ksize, sigma=self._corn_sigma
        )
        corn = self._percentile_normalize(corn, self._norm_pct)

        # 8. Weighted combination
        combined = (
            self._w_edge * edge_density
            + self._w_var * local_variance
            + self._w_ent * ent
            + self._w_corn * corn
        )
        combined = np.clip(combined, 0.0, 1.0).astype(np.float32)

        return {
            'edge': edge_density.astype(np.float32),
            'var': local_variance.astype(np.float32),
            'ent': ent,
            'corn': corn,
            'combined': combined,
        }

    def compute_combined(self, image_rgb: np.ndarray) -> np.ndarray:
        """Returns only the combined heatmap [H, W] float32.

        Convenience wrapper for callers that do not need individual components.
        """
        return self.compute(image_rgb)['combined']

    def batch_compute(self, images: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Compute heatmaps for a list of RGB images.

        Args:
            images: List of RGB images, each (H_i, W_i, 3), dtype uint8.

        Returns:
            List of dicts, each with ``'edge'``, ``'var'``, ``'ent'``, ``'corn'``,
            ``'combined'``.
        """
        return [self.compute(img) for img in images]
