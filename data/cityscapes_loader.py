"""Cityscapes dataset loader for UDA semantic segmentation.

Supports two modes:
- Train split: returns (image, heatmap) with precomputed structural heatmaps.
- Val split: returns (image, label) with ground-truth labels for evaluation.
- skip_heatmap flag allows disabling heatmap loading (baseline_mean_teacher).

Construction accepts either explicit keyword arguments or an OmegaConf config
object as the first positional argument (used by the training loop).
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset

from data.augmentations import TargetAugment

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_LABEL_SUFFIX = "_gtFine_labelTrainIds.png"
_IMG_SUFFIX = "_leftImg8bit.png"

# Default structural-prior weights (match configs/default.yaml)
_DEFAULT_SP_WTS = OmegaConf.create({
    "structural_prior": {
        "edge_weight": 0.25,
        "variance_weight": 0.25,
        "entropy_weight": 0.25,
        "cornerness_weight": 0.25,
    }
})


class CityscapesDataset(Dataset):
    """Cityscapes dataset for UDA training and evaluation.

    For the **train** split returns ``(image, heatmap)`` where heatmap
    is a structural complexity map loaded from four precomputed component
    files (edge, var, ent, corn) and combined at load-time using config
    weights, or an all-ones dummy if ``skip_heatmap=True``.
    For the **val** split returns ``(image, label)`` with ground-truth trainIDs.

    Args:
        root: Root directory of Cityscapes dataset (leftImg8bit + gtFine).
        split: ``"train"`` or ``"val"``.
        heatmap_root: Path to directory containing precomputed .npy component
            files mirroring the leftImg8bit/train structure. Required when
            ``split="train"`` and ``skip_heatmap=False``.
        crop_size: Target (H, W). If ``None``, original image size is kept.
        augment: Whether to apply target-domain augmentation (flip + crop).
        skip_heatmap: If ``True``, return an all-ones dummy heatmap instead
            of loading from disk (used by source_only and mean_teacher baselines).
        cfg: OmegaConf config with a ``structural_prior`` section containing
            ``edge_weight``, ``variance_weight``, ``entropy_weight``,
            ``cornerness_weight``.  If ``None``, defaults (0.25 each) are used.
    """

    def __init__(
        self,
        root_or_cfg: Union[str, OmegaConf],
        split: str = "train",
        heatmap_root: Optional[str] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        skip_heatmap: bool = False,
        cfg: Optional[OmegaConf] = None,
    ) -> None:
        super().__init__()
        # Support OmegaConf DictConfig as first argument (training loop usage).
        if isinstance(root_or_cfg, DictConfig):
            _cfg = root_or_cfg
            root = str(_cfg.training.target_root)
            heatmap_root = _cfg.training.get("heatmap_root", None)
            crop_size = tuple(_cfg.training.crop_size) if _cfg.training.crop_size is not None else None
            augment = True
            skip_heatmap = _cfg.training.get("skip_heatmap", False)
            cfg = _cfg  # pass along for structural_prior weights
        else:
            root = root_or_cfg

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.skip_heatmap = skip_heatmap

        # Heatmap combination weights
        if cfg is not None:
            self._sp_cfg = cfg.structural_prior
        else:
            self._sp_cfg = _DEFAULT_SP_WTS.structural_prior

        img_dir = self.root / "leftImg8bit" / split

        if split == "train":
            # For training, discover images; heatmaps loaded on demand
            self._samples: list[Path] = []
            for img_path in sorted(img_dir.rglob(f"*{_IMG_SUFFIX}")):
                self._samples.append(img_path)

            if len(self._samples) == 0:
                raise FileNotFoundError(f"No training images found in {img_dir}")

            if heatmap_root is not None:
                self._heatmap_root = Path(heatmap_root)
            else:
                self._heatmap_root = None
        else:
            # For validation, discover image-label pairs
            label_dir = self.root / "gtFine" / split
            self._samples: list[Tuple[Path, Path]] = []
            for img_path in sorted(img_dir.rglob(f"*{_IMG_SUFFIX}")):
                stem = img_path.name[: -len(_IMG_SUFFIX)]
                label_path = (
                    label_dir / img_path.relative_to(img_dir).parent / f"{stem}{_LABEL_SUFFIX}"
                )
                if label_path.exists():
                    self._samples.append((img_path, label_path))

            if len(self._samples) == 0:
                raise FileNotFoundError(
                    f"No validation images found in {img_dir} "
                    f"with matching labels in {label_dir}"
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split == "train":
            return self._get_train_item(idx)
        else:
            return self._get_val_item(idx)

    def _load_and_combine_heatmap(self, img_path: Path) -> np.ndarray:
        """Load 4 component heatmap files and combine using config weights.

        The combination is done at load time so weights can be changed via
        config without re-running precomputation.

        Args:
            img_path: Path to the source ``_leftImg8bit.png`` image.

        Returns:
            Combined heatmap (H, W), float32, values in [0.0, 1.0].
        """
        stem = img_path.name[: -len(_IMG_SUFFIX)]

        # Compute base directory for heatmap files
        if self._heatmap_root is not None:
            rel = img_path.relative_to(img_path.parent.parent.parent)
            heat_dir = self._heatmap_root / rel.parent.parent / rel.parent.name
        else:
            heat_dir = img_path.parent

        # Check all 4 component files exist
        comp_suffixes = {
            'edge': '_satg_edge.npy',
            'var': '_satg_var.npy',
            'ent': '_satg_ent.npy',
            'corn': '_satg_corn.npy',
        }
        missing = []
        for suffix in comp_suffixes.values():
            path = heat_dir / f"{stem}{suffix}"
            if not path.exists():
                missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                f"Missing heatmap component(s) for {img_path}.\n"
                f"Expected: *_satg_edge.npy, *_satg_var.npy, "
                f"*_satg_ent.npy, *_satg_corn.npy\n"
                f"Not found:\n" + "\n".join(missing)
            )

        # Load components
        components = {}
        for key, suffix in comp_suffixes.items():
            path = heat_dir / f"{stem}{suffix}"
            components[key] = np.load(str(path)).astype(np.float32)

        # Combine with config weights
        sp = self._sp_cfg
        H = (
            sp.edge_weight * components['edge']
            + sp.variance_weight * components['var']
            + sp.entropy_weight * components['ent']
            + sp.cornerness_weight * components['corn']
        )
        return np.clip(H, 0.0, 1.0).astype(np.float32)

    def _get_train_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self._samples[idx]

        # --- Load image (BGR → RGB float32) ---
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # --- Load heatmap (dummy when skipped; real otherwise) ---
        # NOTE: the crop/augment below MUST run in both cases, otherwise
        # skip_heatmap configs (source_only, mean_teacher) would train on the
        # full-resolution target while everything else uses crop_size — an 8x
        # cost and an unintended resolution mismatch vs. the config.
        if self.skip_heatmap:
            heatmap = np.zeros(img_rgb.shape[:2], dtype=np.float32)
        else:
            heatmap = self._load_and_combine_heatmap(img_path)

        # --- Augmentation (spatial only: flip + crop) ---
        if self.augment:
            aug = TargetAugment(crop_size=self.crop_size)
            img_rgb, heatmap = aug(img_rgb, heatmap)
        elif self.crop_size is not None:
            h, w = self.crop_size
            img_rgb = _centre_crop(img_rgb, h, w)
            heatmap = _centre_crop(heatmap, h, w)

        # --- Normalise image ---
        img_rgb = (img_rgb - _MEAN) / _STD

        # --- Convert to tensors ---
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        if self.skip_heatmap:
            return img_tensor, torch.zeros(1)
        heatmap_tensor = torch.from_numpy(heatmap).float()

        return img_tensor, heatmap_tensor

    def _get_val_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self._samples[idx]

        # --- Load image (BGR → RGB float32) ---
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # --- Load label (single-channel, already trainIDs) ---
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"Failed to load label: {label_path}")

        # --- Centre crop for consistent evaluation ---
        if self.crop_size is not None:
            h, w = self.crop_size
            img_rgb = _centre_crop(img_rgb, h, w)
            label = _centre_crop(label, h, w)

        # --- Normalise image ---
        img_rgb = (img_rgb - _MEAN) / _STD

        # --- Convert to tensors ---
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label).long()

        return img_tensor, label_tensor


def _centre_crop(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Centre-crop a 2D or 3D numpy array."""
    cur_h, cur_w = arr.shape[:2]
    y_start = max(0, (cur_h - h) // 2)
    x_start = max(0, (cur_w - w) // 2)
    return arr[y_start : y_start + h, x_start : x_start + w]
