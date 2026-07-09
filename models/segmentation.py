"""SegmentationModel — DeepLabV3+ ResNet50 wrapper for 19-class Cityscapes.

Wraps torchvision's deeplabv3_resnet50 pretrained on COCO, replaces the final
classifier and aux-classifier heads for 19-class output, and provides a simple
forward interface returning (main_logits, aux_logits).
"""

from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class SegmentationModel(nn.Module):
    """DeepLabV3+ ResNet50 wrapper producing 19-class Cityscapes logits.

    The underlying ``_model`` is a pre-trained DeepLabV3 with the classifier
    heads replaced to output ``num_classes`` instead of the default 21 (COCO).

    Forward returns a ``(main_logits, aux_logits)`` tuple, both shaped
    ``[B, num_classes, H/8, W/8]``.
    """

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()

        import torchvision.models.segmentation as segmentation_models

        self._model = segmentation_models.deeplabv3_resnet50(
            weights="DEFAULT",
            weights_backend=None,  # suppress unused-keyword warning
        )

        # Replace classifier head: default is Conv2d(256, 21, 1) → Conv2d(256, 19, 1)
        self._model.classifier[-1] = nn.Conv2d(256, num_classes, 1)

        # Replace aux classifier head: default is Conv2d(256, 21, 1) → Conv2d(256, 19, 1)
        self._model.aux_classifier[-1] = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor, shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(out, aux)`` where:
                - ``out``: main logits, shape ``(B, num_classes, H/8, W/8)``
                - ``aux``: auxiliary logits, shape ``(B, num_classes, H/8, W/8)``
        """
        result = self._model(x)

        # Downsample to 1/8 of input resolution for SATG loss computation.
        # torchvision's DeepLabV3 interpolates classifier output back to
        # input resolution; we remove that upsampling here for efficiency
        # and to match the expected operating resolution.
        h, w = x.shape[2] // 8, x.shape[3] // 8
        out = F.interpolate(result["out"], size=(h, w), mode="bilinear", align_corners=False)
        aux = F.interpolate(result["aux"], size=(h, w), mode="bilinear", align_corners=False)

        return out, aux
