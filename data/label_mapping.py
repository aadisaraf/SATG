"""GTA5 → Cityscapes 19-class label mapping.

GTA5 provides 33 semantic class IDs (0–32) with RGB palette-encoded labels.
GTA5 classes 0–18 use the SAME RGB palette as Cityscapes train IDs 0–18.
Classes 19–32 are GTA5-only and map to ignore (255).

Reference: AdaptSegNet / DAFormer convention.
"""

from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# GTA5 class ID → Cityscapes train ID
# ---------------------------------------------------------------------------
# GTA5 classes 0–18 map 1:1 to Cityscapes train IDs 0–18.
# GTA5-only classes 19–32 are mapped to 255 (ignore).
GTA5_TO_CITYSCAPES: Dict[int, int] = {
    0: 0,  # road
    1: 1,  # sidewalk
    2: 2,  # building
    3: 3,  # wall
    4: 4,  # fence
    5: 5,  # pole
    6: 6,  # traffic light
    7: 7,  # traffic sign
    8: 8,  # vegetation
    9: 9,  # terrain
    10: 10,  # sky
    11: 11,  # person
    12: 12,  # rider
    13: 13,  # car
    14: 14,  # truck
    15: 15,  # bus
    16: 16,  # train
    17: 17,  # motorcycle
    18: 18,  # bicycle
    19: 255,  # GTA5-only → ignore
    20: 255,
    21: 255,
    22: 255,
    23: 255,
    24: 255,
    25: 255,
    26: 255,
    27: 255,
    28: 255,
    29: 255,
    30: 255,
    31: 255,
    32: 255,
}

# ---------------------------------------------------------------------------
# Cityscapes palette → trainID lookup
# ---------------------------------------------------------------------------
# GTA5 uses the same RGB colours as Cityscapes for classes 0–18.
_CITYSCAPES_PALETTE: Dict[Tuple[int, int, int], int] = {
    (128, 64, 128): 0,  # road
    (244, 35, 232): 1,  # sidewalk
    (70, 70, 70): 2,  # building
    (102, 102, 156): 3,  # wall
    (190, 153, 153): 4,  # fence
    (153, 153, 153): 5,  # pole
    (250, 170, 30): 6,  # traffic light
    (220, 220, 0): 7,  # traffic sign
    (107, 142, 35): 8,  # vegetation
    (152, 251, 152): 9,  # terrain
    (70, 130, 180): 10,  # sky
    (220, 20, 60): 11,  # person
    (255, 0, 0): 12,  # rider
    (0, 0, 142): 13,  # car
    (0, 0, 70): 14,  # truck
    (0, 60, 100): 15,  # bus
    (0, 80, 100): 16,  # train
    (0, 0, 230): 17,  # motorcycle
    (119, 11, 32): 18,  # bicycle
}

# Combined lookup: RGB → Cityscapes trainID.
# GTA5 colors not in this palette → 255 by default.
RGB_TO_TRAINID: Dict[Tuple[int, int, int], int] = dict(_CITYSCAPES_PALETTE)


def map_gta5_label(label_rgb: np.ndarray) -> np.ndarray:
    """Convert a GTA5 RGB label image to Cityscapes 19-class train IDs.

    Args:
        label_rgb: GTA5 label image, shape (H, W, 3), dtype uint8,
                   encoded with the Cityscapes-compatible GTA5 palette.

    Returns:
        Cityscapes trainID map, shape (H, W), dtype uint8,
        values in {0..18} ∪ {255}.

    Raises:
        ValueError: If input is not uint8 or does not have 3 channels.
    """
    if label_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input, got {label_rgb.dtype}")
    if label_rgb.ndim != 3 or label_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB input, got shape {label_rgb.shape}")

    h, w = label_rgb.shape[:2]
    flat = label_rgb.reshape(-1, 3)
    out = np.full(h * w, 255, dtype=np.uint8)

    for (r, g, b), train_id in RGB_TO_TRAINID.items():
        mask = (flat[:, 0] == r) & (flat[:, 1] == g) & (flat[:, 2] == b)
        out[mask] = train_id

    return out.reshape(h, w)
