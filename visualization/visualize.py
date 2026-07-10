"""Visualization CLI — generates 1×5 panel trust-mask visualisations.

Usage::

    python -m visualization.visualize \\
        --checkpoint checkpoints/satg_hard_seed42/last.pth \\
        --config configs/satg_hard.yaml \\
        [--num_images 10] \\
        [--output_dir visualizations/]

Panels
------
1. Input Image (unnormalised RGB)
2. Teacher confidence map (viridis)
3. Structural heatmap (viridis)
4. Trust mask overlay (accepted = coloured, rejected = greyed out)
5. SATG-filtered pseudo-label (Cityscapes colour palette, 19 classes)

# allow: SIZE_OK — CLI entry point with multiple helpers
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, must come before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.cityscapes_loader import CityscapesDataset  # noqa: E402
from models.segmentation import SegmentationModel  # noqa: E402
from satg.structural_prior import StructuralPrior  # noqa: E402
from satg.trust_gate import HardTrustGate, SoftWeightTrustGate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalisation stats (used by CityscapesDataset)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cityscapes 19-class colour palette (train ID → RGB)
CITYSCAPES_PALETTE: dict = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list (used by tests).  Defaults to ``sys.argv``.

    Returns:
        Parsed namespace with fields ``checkpoint``, ``config``, ``num_images``,
        and ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Generate 1×5 panel trust-mask visualisations for SATG."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the variant config YAML (e.g. configs/satg_hard.yaml).",
    )
    parser.add_argument(
        "--num_images",
        default=10,
        type=int,
        help="Number of validation images to visualise (default: 10).",
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations/",
        type=str,
        help="Root output directory (default: visualizations/). "
        "Per-config subdirectories are created automatically.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def unnormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised [C, H, W] tensor back to a [H, W, 3] uint8 image.

    Args:
        img_tensor: Float tensor of shape ``(C, H, W)`` with ImageNet
            normalisation (mean = 0.485, 0.456, 0.406; std = 0.229, 0.224, 0.225).

    Returns:
        RGB ``uint8`` array of shape ``(H, W, 3)`` with values in ``[0, 255]``.
    """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * _STD + _MEAN  # unnormalise
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def colorize_label(label: np.ndarray) -> np.ndarray:
    """Map a Cityscapes train-ID label map to an RGB colour image.

    Pixels with values outside ``0..18`` (e.g. ignore index 255) are shown
    as black ``(0, 0, 0)``.

    Args:
        label: Integer array of shape ``(H, W)`` with Cityscapes train IDs.

    Returns:
        ``uint8`` RGB array of shape ``(H, W, 3)``.
    """
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in CITYSCAPES_PALETTE.items():
        mask = label == cls_id
        rgb[mask] = colour
    return rgb


def create_trust_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a trust-mask overlay visualisation.

    Pixels where ``mask >= 0.5`` retain their original colour (trusted).
    Pixels where ``mask < 0.5`` are greyed out (rejected / low-weight).

    Args:
        image: RGB ``uint8`` array of shape ``(H, W, 3)``.
        mask: Float array of shape ``(H, W)`` with values in ``[0, 1]``.

    Returns:
        ``uint8`` overlay image, same shape as *image*.
    """
    overlay = image.copy().astype(np.float32)
    grey = np.mean(image, axis=2, keepdims=True)  # luminance (H, W, 1)
    rejected = mask < 0.5
    # Greyscale the rejected regions
    overlay[rejected] = grey[rejected] * np.ones(3, dtype=np.float32)
    return overlay.astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure creation
# ---------------------------------------------------------------------------


def create_figure(
    image_rgb: np.ndarray,
    confidence: np.ndarray,
    struct_heatmap: np.ndarray,
    trust_mask: np.ndarray,
    pseudo_label: np.ndarray,
    gate_type: str,
) -> plt.Figure:
    """Create a 1×5 panel figure for one validation image.

    Panels
    ------
    1. Input RGB image.
    2. Teacher confidence map (viridis, 0 = dark → 1 = bright).
    3. Structural heatmap (viridis, 0 = dark → 1 = bright).
    4. Trust mask overlay (trusted = colour, rejected = grey).
    5. SATG-filtered pseudo-label (Cityscapes colour palette).

    Args:
        image_rgb: RGB ``uint8`` image, shape ``(H, W, 3)``.
        confidence: Confidence map, shape ``(H, W)``, values in ``[0, 1]``.
        struct_heatmap: Structural prior map, shape ``(H, W)``, values in ``[0, 1]``.
        trust_mask: Trust mask or weights, shape ``(H, W)``, values in ``[0, 1]``.
        pseudo_label: Pseudo-label train IDs, shape ``(H, W)``, ``uint8``.
        gate_type: Trust gate type name for the panel 4 title.

    Returns:
        The matplotlib ``Figure`` with 5 subplots arranged in a 1×5 grid.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Panel 1 — Input image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Panel 2 — Teacher confidence
    axes[1].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("Teacher Confidence")
    axes[1].axis("off")

    # Panel 3 — Structural heatmap
    axes[2].imshow(struct_heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Structural Heatmap")
    axes[2].axis("off")

    # Panel 4 — Trust mask overlay
    overlay = create_trust_overlay(image_rgb, trust_mask)
    axes[3].imshow(overlay)
    axes[3].set_title(f"Trust Mask ({gate_type})")
    axes[3].axis("off")

    # Panel 5 — SATG-filtered pseudo-label
    label_colour = colorize_label(pseudo_label)
    # Mask out rejected regions (black)
    label_colour[trust_mask < 0.5] = 0
    axes[4].imshow(label_colour)
    axes[4].set_title("SAT Pseudo-Label")
    axes[4].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — loads checkpoint, runs inference, generates panels.

    The pipeline:
    1. Parses CLI arguments.
    2. Loads and merges configs (default + variant).
    3. Loads the model checkpoint (handles both ``model_state`` key and
       direct state dict).
    4. Creates the Cityscapes validation dataset.
    5. Iterates over images, computing teacher confidence, structural priors,
       trust masks, and pseudo-labels for each, and saves 1×5 panel PNGs.
    """
    args = _parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    base_cfg = OmegaConf.load("configs/default.yaml")
    variant_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, variant_cfg)
    config_name = Path(args.config).stem

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ─────────────────────────────────────────────────────────────
    model = SegmentationModel(num_classes=cfg.model.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint  # direct state dict

    # Strip 'module.' prefix from DataParallel-wrapped keys if present
    cleaned: dict = {}
    for k, v in state_dict.items():
        key = k[len("module."):] if k.startswith("module.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = CityscapesDataset(cfg, split="val")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.training.get("num_workers", 2),
        pin_memory=True,
    )

    # ── Structural prior ──────────────────────────────────────────────────
    struct_prior = StructuralPrior(cfg)

    # ── Trust gate ─────────────────────────────────────────────────────────
    gate_type = cfg.trust_gate.type
    if gate_type == "hard":
        gate = HardTrustGate(cfg)
    elif gate_type == "soft_weight":
        gate = SoftWeightTrustGate(cfg)
    else:
        # For soft_label and other types, use hard gate as fallback for
        # visualisation purposes
        gate = HardTrustGate(cfg)
        gate_type = "hard"  # normalise display name

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ────────────────────────────────────────────────────
    for idx, (img_tensor, label_tensor) in enumerate(
        tqdm(loader, desc="Generating visualisations", total=min(args.num_images, len(loader)))
    ):
        if idx >= args.num_images:
            break

        img_tensor = img_tensor.to(device)

        # ── Teacher forward ────────────────────────────────────────────
        with torch.no_grad():
            main_logits, _ = model(img_tensor)

        # Interpolate to label resolution
        target_h, target_w = label_tensor.shape[-2:]
        logits_up = F.interpolate(
            main_logits, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits_up, dim=1)  # [1, 19, H, W]
        confidence = probs.max(dim=1).values  # [1, H, W]
        pseudo_labels = probs.argmax(dim=1)  # [1, H, W]

        # ── Unnormalise for visualisation & structural prior ────────────
        img_rgb = unnormalize(img_tensor[0])  # (H, W, 3), uint8

        # ── Structural prior (computed on the fly) ──────────────────────
        struct_result = struct_prior.compute(img_rgb)
        struct_heatmap = struct_result["combined"]  # (H, W), float32

        # ── Trust mask ──────────────────────────────────────────────────
        conf_t = confidence[0:1]  # keep batch dim
        struct_t = (
            torch.from_numpy(struct_heatmap).unsqueeze(0).float()
        )  # [1, H, W]

        if gate_type == "hard":
            tw = gate.compute_mask(conf_t, struct_t)
        else:  # soft_weight
            tw = gate.compute_weights(conf_t, struct_t)
        tw_np = tw[0].cpu().numpy()  # (H, W)

        # ── Create figure ───────────────────────────────────────────────
        fig = create_figure(
            image_rgb=img_rgb,
            confidence=confidence[0].cpu().numpy(),
            struct_heatmap=struct_heatmap,
            trust_mask=tw_np,
            pseudo_label=pseudo_labels[0].cpu().numpy().astype(np.uint8),
            gate_type=gate_type,
        )

        # ── Save ────────────────────────────────────────────────────────
        stem = f"vis_{idx:04d}"
        save_path = out_dir / f"{stem}.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
