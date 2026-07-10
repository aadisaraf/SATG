"""GTA5 → Cityscapes 19-class label mapping.

GTA5 provides 35 semantic class indices (0–34) encoded as single-channel
indexed PNGs (NOT RGB palette). This module maps GTA5 class indices to
Cityscapes 19-class trainIDs (0–18) or 255 (ignore).

The standard AdaptSegNet / DAFormer convention is used for the mapping.
"""

from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# GTA5 class index → Cityscapes train ID  (35 entries, indices 0–34)
# ---------------------------------------------------------------------------
# Standard AdaptSegNet mapping (GTA5 dataset has 35 classes, 0–34):
#   GTA5 0 (unlabeled)     → 255
#   GTA5 1 (ego vehicle)   → 255
#   GTA5 2 (rect border)   → 255
#   GTA5 3 (out of roi)    → 255
#   GTA5 4 (static)        → 255
#   GTA5 5 (dynamic)       → 255
#   GTA5 6 (ground)        → 255
#   GTA5 7 (road)          →   0
#   GTA5 8 (sidewalk)      →   1
#   GTA5 9 (building)      →   2
#   GTA5 10 (wall)         →   3
#   GTA5 11 (fence)        →   4
#   GTA5 12 (pole)         →   5
#   GTA5 13 (traffic light) →  6
#   GTA5 14 (traffic sign)  →  7
#   GTA5 15 (vegetation)   →   8
#   GTA5 16 (terrain)      →   9
#   GTA5 17 (sky)          →  10
#   GTA5 18 (person)       →  11
#   GTA5 19 (rider)        →  12
#   GTA5 20 (car)          →  13
#   GTA5 21 (truck)        →  14
#   GTA5 22 (bus)          →  15
#   GTA5 23 (train)        →  16
#   GTA5 24 (motorcycle)   →  17
#   GTA5 25 (bicycle)      →  18
#   GTA5 26–34 (unused)    → 255
GTA5_TO_CITYSCAPES_19: Dict[int, int] = {
    0: 255,  # unlabeled → ignore
    1: 255,  # ego vehicle → ignore
    2: 255,  # rect border → ignore
    3: 255,  # out of roi → ignore
    4: 255,  # static → ignore
    5: 255,  # dynamic → ignore
    6: 255,  # ground → ignore
    7: 0,    # road
    8: 1,    # sidewalk
    9: 2,    # building
    10: 3,   # wall
    11: 4,   # fence
    12: 5,   # pole
    13: 6,   # traffic light
    14: 7,   # traffic sign
    15: 8,   # vegetation
    16: 9,   # terrain
    17: 10,  # sky
    18: 11,  # person
    19: 12,  # rider
    20: 13,  # car
    21: 14,  # truck
    22: 15,  # bus
    23: 16,  # train
    24: 17,  # motorcycle
    25: 18,  # bicycle
    26: 255,
    27: 255,
    28: 255,
    29: 255,
    30: 255,
    31: 255,
    32: 255,
    33: 255,
    34: 255,
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


# Backward-compatibility alias (legacy name used in earlier code).
GTA5_TO_CITYSCAPES: Dict[int, int] = GTA5_TO_CITYSCAPES_19
