"""Tests for SATG trainer (RED phase — all should fail until ``SATGTrainer`` class exists).

The training module currently exposes only a ``main()`` function.  These tests
expect a ``SATGTrainer`` class (with ``__init__``, ``train_step``, ``update_ema``,
and logging methods) that does not exist yet.
"""

from pathlib import Path

import pytest
import torch


# ===========================================================================
# T028: Trainer class tests
# ===========================================================================


class TestSATGTrainer:
    """Expected ``SATGTrainer`` class-based trainer API tests.

    The current implementation has a procedural ``main()`` function.  These
    tests define the class-based interface that should eventually replace or
    wrap it.
    """

    def _trainer(self, cfg):
        """Import and instantiate the non-existent ``SATGTrainer`` — raises ImportError."""
        from training.trainer import SATGTrainer  # noqa — expected ImportError
        return SATGTrainer(cfg)

    def test_init_all_components(self, monkeypatch):
        """``SATGTrainer(cfg)`` initialises all components without GPU."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg()
        trainer = self._trainer(cfg)

        assert hasattr(trainer, "student")
        assert hasattr(trainer, "ema")
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "scheduler")
        assert hasattr(trainer, "source_loader")
        assert hasattr(trainer, "target_loader")
        assert hasattr(trainer, "val_loader")

    def test_ema_initialised_and_eval(self, monkeypatch):
        """After init, ``trainer.ema.model`` is in ``eval`` mode with ``requires_grad=False``."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg()
        trainer = self._trainer(cfg)

        assert hasattr(trainer.ema, "model")
        assert not trainer.ema.model.training
        for p in trainer.ema.model.parameters():
            assert not p.requires_grad

    def test_one_training_step_executes(self, monkeypatch):
        """``trainer.train_step()`` runs one iteration without raising."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg(num_iters=1)
        trainer = self._trainer(cfg)

        trainer.source_loader = _fake_loader(batch_size=2)
        trainer.target_loader = _fake_loader(batch_size=2, is_target=True)

        loss = trainer.train_step()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_ema_update_after_step(self, monkeypatch):
        """After ``train_step()``, EMA teacher parameters change."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg(num_iters=1)
        trainer = self._trainer(cfg)
        trainer.source_loader = _fake_loader(batch_size=2)
        trainer.target_loader = _fake_loader(batch_size=2, is_target=True)

        ema_before = {n: p.data.clone() for n, p in trainer.ema.model.named_parameters()}
        trainer.train_step()
        ema_after = {n: p.data.clone() for n, p in trainer.ema.model.named_parameters()}

        diffs = [n for n in ema_before if not torch.equal(ema_before[n], ema_after[n])]
        assert len(diffs) > 0, "Expected at least one EMA parameter to change"

    def test_loss_logging_occurs(self, monkeypatch):
        """After ``train_step()``, loss is recorded."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg(num_iters=1)
        trainer = self._trainer(cfg)
        trainer.source_loader = _fake_loader(batch_size=2)
        trainer.target_loader = _fake_loader(batch_size=2, is_target=True)

        trainer.train_step()

        if hasattr(trainer, "loss_history"):
            assert len(trainer.loss_history) >= 1
        elif hasattr(trainer, "log"):
            pass  # accept .log() convention too
        else:
            pytest.fail("Trainer must expose either .loss_history or a .log() method")

    def test_student_in_train_mode(self, monkeypatch):
        """After init the student model is in ``train`` mode."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg()
        trainer = self._trainer(cfg)
        assert trainer.student.training

    def test_optimizer_configured(self, monkeypatch):
        """``trainer.optimizer`` is a ``torch.optim.Optimizer``."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg()
        trainer = self._trainer(cfg)
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_scheduler_configured(self, monkeypatch):
        """``trainer.scheduler`` is a ``LRScheduler``."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = _make_minimal_cfg()
        trainer = self._trainer(cfg)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LRScheduler)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_minimal_cfg(num_iters: int = 2):
    """Return an OmegaConf config with minimal trainer-compatible settings."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "seed": 42,
        "training": {
            "crop_size": [64, 64],
            "batch_size": 2,
            "num_workers": 0,
            "iterations": num_iters,
            "eval_interval": 10,
            "lr": 1e-3,
            "backbone_lr_multiplier": 0.1,
            "optimizer_momentum": 0.9,
            "weight_decay": 1e-4,
            "poly_power": 0.9,
            "lambda_target": 0.1,
            "aux_loss_weight": 0.4,
            "gradient_clip_norm": 5.0,
            "use_amp": False,
            "rare_class_sampling": False,
            "target_root": str(Path("/tmp/fake_cityscapes")),
            "source_root": str(Path("/tmp/fake_gta5")),
            "heatmap_root": None,
            "skip_heatmap": True,
        },
        "model": {
            "backbone": "resnet101",
            "num_classes": 19,
        },
        "ema": {
            "momentum": 0.999,
        },
        "trust_gate": {
            "type": "hard",
            "tau_conf": 0.90,
            "tau_struct": 0.60,
        },
        "logging": {
            "backend": "none",
            "log_every": 1,
            "project": "test",
        },
        "checkpoint": {
            "save_dir": str(Path("/tmp/test_trainer_ckpt")),
        },
    })


def _fake_loader(batch_size: int = 2, is_target: bool = False):
    """Return a DataLoader that yields synthetic batches."""
    from torch.utils.data import DataLoader, TensorDataset

    imgs = torch.randn(batch_size, 3, 64, 64)
    if is_target:
        second = torch.rand(batch_size, 64, 64)
    else:
        second = torch.randint(0, 19, (batch_size, 64, 64), dtype=torch.long)

    dataset = TensorDataset(imgs, second)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
