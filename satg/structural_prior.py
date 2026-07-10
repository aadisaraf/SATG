"""StructuralPrior — classical-CV structural complexity heatmap.

Computes a per-pixel structural complexity score in [0, 1] using edge density
and local variance.  No learned / ML parameters — pure OpenCV + NumPy.
"""

from typing import List

import cv2
import numpy as np
from omegaconf import OmegaConf


class StructuralPrior:
    """Compute structural complexity heatmaps from RGB images.

    Config keys (under ``structural_prior``):

        edge_low_threshold (int, default 50)
            Lower hysteresis threshold for Canny.
        edge_high_threshold (int, default 150)
            Upper hysteresis threshold for Canny.
        gaussian_sigma (float, default 2.0)
            Sigma for Gaussian blur pre-filter.
        edge_kernel_size (int, default 15)
            Side length of the uniform kernel for edge density.
        variance_kernel_size (int, default 15)
            Side length of the uniform kernel for local variance.
        edge_weight (float, default 0.5)
            Linear weight for the edge-density term.
        variance_weight (float, default 0.5)
            Linear weight for the local-variance term.
    """

    def __init__(self, cfg: OmegaConf) -> None:
        sc = cfg.structural_prior
        self._edge_low = sc.edge_low_threshold
        self._edge_high = sc.edge_high_threshold
        self._gauss_sigma = sc.gaussian_sigma
        self._edge_ksize = sc.edge_kernel_size
        self._var_ksize = sc.variance_kernel_size
        self._w_edge = sc.edge_weight
        self._w_var = sc.variance_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, image_rgb: np.ndarray) -> np.ndarray:
        """Compute structural complexity heatmap for a single RGB image.

        Args:
            image_rgb: RGB image, shape (H, W, 3), dtype uint8, values [0, 255].

        Returns:
            Heatmap, shape (H, W), dtype float32, values [0.0, 1.0].
        """
        # 1. RGB → grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # (H, W), uint8

        # 2. Gaussian blur
        blurred = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=self._gauss_sigma)

        # 3. Canny edge detection
        edges = cv2.Canny(blurred, self._edge_low, self._edge_high)  # (H, W), uint8 {0, 255}

        # 4. Edge density via uniform kernel convolution → [0, 1]
        edge_kernel = np.ones((self._edge_ksize, self._edge_ksize), dtype=np.float32) / (
            self._edge_ksize * self._edge_ksize
        )
        edge_density = cv2.filter2D(edges.astype(np.float32), ddepth=-1, kernel=edge_kernel)
        edge_density = np.clip(edge_density / 255.0, 0.0, 1.0)  # normalise from {0,255} to [0,1]

        # 5. Local variance: var = blur(I²) - blur(I)²
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (self._var_ksize, self._var_ksize))
        mean_sq = cv2.blur(gray_f ** 2, (self._var_ksize, self._var_ksize))
        variance = mean_sq - mean ** 2
        # Normalise variance to [0, 1] via min-max (clamp at 0 to avoid tiny negatives)
        v_min, v_max = variance.min(), variance.max()
        if v_max > v_min:
            local_variance = (variance - v_min) / (v_max - v_min)
        else:
            local_variance = np.zeros_like(variance)

        # 6. Weighted combination
        heatmap = self._w_edge * edge_density + self._w_var * local_variance

        # 7. Clip to [0, 1] and return float32
        return np.clip(heatmap, 0.0, 1.0).astype(np.float32)

    def batch_compute(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Compute heatmaps for a list of RGB images.

        Args:
            images: List of RGB images, each (H_i, W_i, 3), dtype uint8.

        Returns:
            List of heatmaps, each (H_i, W_i), dtype float32.
        """
        return [self.compute(img) for img in images]