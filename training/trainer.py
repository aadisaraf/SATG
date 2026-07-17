"""Main UDA training loop for Structure-Aware Trust Gating (SATG).

Usage::

    python -m training.trainer --config configs/satg_hard.yaml
    python -m training.trainer --config configs/satg_soft_weight.yaml
    python -m training.trainer --config configs/satg_soft_label.yaml

CLI overrides (dot-list notation)::

    python -m training.trainer --config configs/satg_hard.yaml \\
        training.iterations=10 training.eval_interval=5
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data.gta5_loader import GTA5Dataset
from data.cityscapes_loader import CityscapesDataset
from evaluation.evaluator import compute_miou
from models.segmentation import SegmentationModel
from models.ema import EMAModel
from satg.trust_gate import HardTrustGate, SoftWeightTrustGate
from satg.soft_label import TemperatureSoftLabel
from satg.losses import SATGLoss, SoftLabelKLLoss


def main() -> None:
    # ── Argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help=(
            "Override the run identity (checkpoint dir + wandb name). Defaults "
            "to '<config-stem>_seed<seed>'. Give distinct names to runs that "
            "share a config but differ by overrides (e.g. ablations) so their "
            "checkpoints don't collide."
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        type=str,
        help=(
            "Resume training. Bare --resume auto-loads <save_dir>/last.pth; "
            "pass a path to resume from a specific checkpoint."
        ),
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # ── Config loading ────────────────────────────────────────────────────
    base_cfg = OmegaConf.load("configs/default.yaml")
    variant_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, variant_cfg)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # ── Seeds ─────────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    # NOTE: benchmark=True + deterministic=True can silently degrade reproducibility
    # on some CUDA versions. Only enable if you understand the interaction.
    torch.backends.cudnn.benchmark = cfg.training.get("cudnn_benchmark", False)

    # ── Logging and checkpoint setup ──────────────────────────────────────
    run_name = args.run_name or f"{Path(args.config).stem}_seed{cfg.seed}"
    save_dir = Path(cfg.checkpoint.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_dir / "config.yaml")

    # ── Resume checkpoint resolution ──────────────────────────────────────
    # Resolve the checkpoint to resume from (if any) and load it onto CPU so
    # its wandb run id is available before wandb.init(). Model/optimizer states
    # are applied further down, once those objects exist.
    resume_ckpt = None
    if args.resume is not None:
        resume_path = (
            save_dir / "last.pth" if args.resume == "auto" else Path(args.resume)
        )
        if resume_path.is_file():
            resume_ckpt = torch.load(resume_path, map_location="cpu")
            print(
                f"Resuming from {resume_path} "
                f"(iteration {resume_ckpt.get('iteration', 0)})"
            )
        elif args.resume == "auto":
            print(f"No checkpoint at {resume_path} — starting fresh.")
        else:
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")

    if cfg.logging.backend == "wandb":
        wandb_run_id = (
            resume_ckpt.get("wandb_run_id") if resume_ckpt else None
        ) or wandb.util.generate_id()
        wandb.init(
            project=cfg.logging.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            id=wandb_run_id,
            resume="allow",
        )
    else:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))

    # Append to metrics.csv when resuming an existing run; otherwise truncate.
    csv_path = save_dir / "metrics.csv"
    resume_csv = resume_ckpt is not None and csv_path.is_file()
    csv_file = open(csv_path, "a" if resume_csv else "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "iteration",
            "total_loss",
            "source_loss",
            "target_loss",
            "trust_coverage_ratio",
            "mean_temperature",
            "lr",
            "val_miou",
        ],
    )
    if not resume_csv:
        csv_writer.writeheader()

    # ── Device and models ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = SegmentationModel(num_classes=cfg.model.num_classes).to(device)
    ema = EMAModel(student, cfg.ema.momentum)
    ema.model = ema.model.to(device)

    # ── Trust gate initialisation (three-way) ─────────────────────────────
    if cfg.trust_gate.type == "hard":
        hard_gate = HardTrustGate(cfg)
        satg_loss = SATGLoss()
    elif cfg.trust_gate.type == "soft_weight":
        soft_weight_gate = SoftWeightTrustGate(cfg)
        satg_loss = SATGLoss()
    elif cfg.trust_gate.type == "soft_label":
        soft_label_mod = TemperatureSoftLabel(cfg)
        soft_label_kl_loss = SoftLabelKLLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────
    backbone_params = [
        p for n, p in student.named_parameters() if "backbone" in n
    ]
    head_params = [
        p for n, p in student.named_parameters() if "backbone" not in n
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": backbone_params,
                "lr": cfg.training.lr * cfg.model.backbone_lr_multiplier,
            },
            {"params": head_params, "lr": cfg.training.lr},
        ],
        momentum=cfg.training.optimizer_momentum,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=cfg.training.iterations,
        power=cfg.training.poly_power,
    )

    scaler = GradScaler(enabled=cfg.training.use_amp)

    # ── Restore state from resume checkpoint ──────────────────────────────
    start_iter = 0
    resumed_best_miou = 0.0
    if resume_ckpt is not None:
        student.load_state_dict(resume_ckpt["model_state"])
        ema.load_state_dict(resume_ckpt["ema_state"])
        ema.model = ema.model.to(device)
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        if resume_ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        start_iter = int(resume_ckpt.get("iteration", 0)) + 1
        resumed_best_miou = float(resume_ckpt.get("best_miou", 0.0))

    # ── Datasets and loaders ──────────────────────────────────────────────
    source_dataset = GTA5Dataset(cfg)
    target_dataset = CityscapesDataset(cfg, split="train")
    val_dataset = CityscapesDataset(cfg, split="val")
    source_loader = DataLoader(
        source_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    source_iter = iter(source_loader)

    def next_source():
        nonlocal source_iter
        try:
            return next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            return next(source_iter)

    target_iter = itertools.cycle(iter(target_loader))

    # ── Source criterion with rare class sampling ─────────────────────────
    if cfg.training.rare_class_sampling:
        class_weights = source_dataset.rare_class_weights.to(device)
        source_criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=255
        )
    else:
        source_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ── Training loop ─────────────────────────────────────────────────────
    best_miou = resumed_best_miou
    mean_temp = 0.0

    pbar = tqdm(
        total=cfg.training.iterations,
        initial=start_iter,
        desc="Training",
        unit="iter",
    )
    for iteration in range(start_iter, cfg.training.iterations):
        student.train()

        # ── Source forward ────────────────────────────────────────────────
        src_imgs, src_labels = next_source()
        src_imgs = src_imgs.to(device)
        src_labels = src_labels.to(device)
        with autocast(device_type="cuda", enabled=cfg.training.use_amp):
            src_main, src_aux = student(src_imgs)
            h, w = src_labels.shape[-2:]
            src_main = F.interpolate(
                src_main, (h, w), mode="bilinear", align_corners=False
            )
            src_aux = F.interpolate(
                src_aux, (h, w), mode="bilinear", align_corners=False
            )
            source_loss = source_criterion(src_main, src_labels) + cfg.training.aux_loss_weight * source_criterion(
                src_aux, src_labels
            )

        # ── Target forward ────────────────────────────────────────────────
        tgt_imgs, tgt_heatmaps = next(target_iter)
        tgt_imgs = tgt_imgs.to(device)
        tgt_heatmaps = tgt_heatmaps.to(device)  # [B, H, W] combined heatmap

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=cfg.training.use_amp):
                tgt_t_main, _ = ema.model(tgt_imgs)
                tgt_h, tgt_w = tgt_heatmaps.shape[-2:]
                tgt_teacher_logits = F.interpolate(
                    tgt_t_main,
                    (tgt_h, tgt_w),
                    mode="bilinear",
                    align_corners=False,
                )
                tgt_probs = F.softmax(tgt_teacher_logits, dim=1)
                confidence = tgt_probs.max(dim=1).values
                pseudo_labels = tgt_probs.argmax(dim=1)

        with autocast(device_type="cuda", enabled=cfg.training.use_amp):
            tgt_s_main, _ = student(tgt_imgs)
            tgt_student_logits = F.interpolate(
                tgt_s_main,
                (tgt_h, tgt_w),
                mode="bilinear",
                align_corners=False,
            )

        # ── Trust gate / target loss ──────────────────────────────────────
        if cfg.trust_gate.type == "hard":
            tw = hard_gate.compute_mask(confidence, tgt_heatmaps)
            target_loss = satg_loss(tgt_student_logits, pseudo_labels, tw)
            trust_coverage = tw.mean().item()

        elif cfg.trust_gate.type == "soft_weight":
            tw = soft_weight_gate.compute_weights(confidence, tgt_heatmaps)
            target_loss = satg_loss(tgt_student_logits, pseudo_labels, tw)
            trust_coverage = tw.mean().item()

        elif cfg.trust_gate.type == "soft_label":
            st = soft_label_mod.compute_soft_targets(
                tgt_teacher_logits, tgt_heatmaps
            )
            cm = soft_label_mod.compute_confidence_mask(
                tgt_teacher_logits, cfg.trust_gate.tau_conf
            )
            target_loss = soft_label_kl_loss(
                tgt_student_logits, st, cm
            )
            trust_coverage = cm.float().mean().item()
            with torch.no_grad():
                T = soft_label_mod.compute_temperature(tgt_heatmaps)
                mean_temp = (
                    T[cm.bool()].mean().item() if cm.sum() > 0 else 0.0
                )

        # ── Backward ──────────────────────────────────────────────────────
        total_loss = source_loss + cfg.training.lambda_target * target_loss
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        if cfg.training.gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=cfg.training.gradient_clip_norm
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(student)
        pbar.update(1)

        # ── Logging ───────────────────────────────────────────────────────
        if iteration % cfg.logging.log_every == 0:
            lr = scheduler.get_last_lr()[-1]
            row = {
                "iteration": iteration,
                "total_loss": round(total_loss.item(), 6),
                "source_loss": round(source_loss.item(), 6),
                "target_loss": round(target_loss.item(), 6),
                "trust_coverage_ratio": round(trust_coverage, 4),
                "mean_temperature": round(mean_temp, 4),
                "lr": round(lr, 8),
                "val_miou": "",
            }
            csv_writer.writerow(row)
            csv_file.flush()
            if cfg.logging.backend == "wandb":
                wandb.log(
                    {k: v for k, v in row.items() if v != ""}, step=iteration
                )

        # ── Evaluation ────────────────────────────────────────────────────
        if iteration % cfg.training.eval_interval == 0 and iteration > 0:
            student.eval()
            val_miou, per_class = compute_miou(student, val_loader, device)
            student.train()
            print(f"[{iteration}] val mIoU: {val_miou:.2f}%")

            csv_writer.writerow(
                {
                    "iteration": iteration,
                    "total_loss": "",
                    "source_loss": "",
                    "target_loss": "",
                    "trust_coverage_ratio": "",
                    "mean_temperature": "",
                    "lr": "",
                    "val_miou": round(val_miou, 4),
                }
            )
            csv_file.flush()
            if cfg.logging.backend == "wandb":
                wandb.log(
                    {
                        "val/miou": val_miou,
                        **{f"val/{k}": v for k, v in per_class.items()},
                    },
                    step=iteration,
                )

            # Update best BEFORE snapshotting so last.pth carries the correct
            # best_miou — this is what --resume reads back.
            is_new_best = val_miou > best_miou
            if is_new_best:
                best_miou = val_miou

            ckpt = {
                "model_state": student.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if cfg.training.use_amp else None,
                "iteration": iteration,
                "best_miou": best_miou,
                "wandb_run_id": (
                    wandb.run.id
                    if cfg.logging.backend == "wandb" and wandb.run is not None
                    else None
                ),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(ckpt, save_dir / "last.pth")
            if is_new_best:
                torch.save(ckpt, save_dir / "best.pth")
                print(f"  ★ New best: {best_miou:.2f}%")

    pbar.close()
    csv_file.close()
    if cfg.logging.backend == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
