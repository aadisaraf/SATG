"""Evaluation pipeline for mIoU and per-class IoU on Cityscapes validation.

The primary entry point is ``compute_miou()``, which runs a model over a
DataLoader and returns the mean IoU and per-class IoU dictionary.
"""

import torch
import torch.nn.functional as F

CITYSCAPES_19 = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def compute_miou(model, dataloader, device, num_classes=19, ignore_index=255):
    """Compute mean IoU and per-class IoU for a segmentation model.

    Args:
        model: nn.Module segmentation model returning ``(main_logits, aux)``.
        dataloader: DataLoader yielding ``(image, label)`` pairs.  Labels are
            expected to be Cityscapes trainIDs in ``{0..18, 255}``.
        device: torch device for inference.
        num_classes: Number of segmentation classes (default 19 for Cityscapes).
        ignore_index: Label value to ignore in evaluation (default 255).

    Returns:
        Tuple of ``(mean_iou, per_class_iou)`` where:
            - ``mean_iou``: float, mean IoU over all classes (percentage).
            - ``per_class_iou``: dict mapping class name to IoU (percentage).
    """
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            main_logits, _ = model(imgs)
            preds = (
                F.interpolate(
                    main_logits,
                    labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                .argmax(1)
            )
            mask = labels != ignore_index
            t = labels[mask].to(torch.int64)
            p = preds[mask].to(torch.int64)
            # Vectorized confusion-matrix accumulation — identical result to the
            # per-pixel Python loop, but ~1000x faster (one bincount vs millions
            # of scalar index-updates).
            idx = t * num_classes + p
            binc = torch.bincount(idx, minlength=num_classes * num_classes)
            confusion += binc.reshape(num_classes, num_classes)

    confusion = confusion.cpu()

    iou = {}
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        d = tp + fp + fn
        iou[CITYSCAPES_19[c]] = (tp / d * 100) if d > 0 else 0.0

    return sum(iou.values()) / num_classes, iou
