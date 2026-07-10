"""Tests for the visualization module (RED phase — tests define expected behaviour)."""

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from omegaconf import OmegaConf

matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def dummy_rgb() -> np.ndarray:
    """Return a small synthetic RGB image (64, 64, 3), uint8."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def dummy_label() -> np.ndarray:
    """Return a synthetic Cityscapes train-ID label map (64, 64), int32."""
    label = np.random.randint(0, 19, (64, 64), dtype=np.uint8)
    return label


@pytest.fixture
def dummy_confidence() -> np.ndarray:
    """Return a synthetic confidence map (64, 64), float32, in [0, 1]."""
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def dummy_trust_mask() -> np.ndarray:
    """Return a synthetic binary trust mask (64, 64), float32, with both 0 and 1."""
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:32, :] = 1.0  # top half trusted, bottom half rejected
    return mask


@pytest.fixture
def dummy_weights() -> np.ndarray:
    """Return a synthetic continuous weight map (64, 64), float32, in [0, 1]."""
    w = np.random.rand(64, 64).astype(np.float32)
    return w


@pytest.fixture
def vis_cfg() -> OmegaConf:
    """Minimal config needed by the visualization helpers."""
    return OmegaConf.create({
        "trust_gate": {
            "type": "hard",
            "tau_conf": 0.90,
            "tau_struct": 0.60,
            "soft_weight_temp_conf": 10.0,
            "soft_weight_temp_struct": 10.0,
            "soft_weight_bias": 0.0,
            "soft_label_k": 4.0,
            "soft_label_t_max": 5.0,
        },
        "structural_prior": {
            "norm_percentile": 95.0,
            "gaussian_sigma": 2.0,
            "edge_kernel_size": 15,
            "variance_kernel_size": 15,
            "edge_weight": 0.25,
            "variance_weight": 0.25,
            "entropy_kernel_radius": 7,
            "entropy_weight": 0.25,
            "cornerness_kernel_size": 15,
            "cornerness_sigma": 2.0,
            "cornerness_weight": 0.25,
        },
        "model": {"num_classes": 19},
        "training": {"target_root": "./data/cityscapes"},
    })


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestUnnormalize:
    """Tests for the image unnormalization helper."""

    def test_output_shape_and_dtype(self):
        """[C, H, W] float32 tensor → [H, W, 3] uint8 array."""
        from visualization.visualize import unnormalize

        tensor = np.random.randn(3, 64, 64).astype(np.float32)
        import torch

        t = torch.from_numpy(tensor)
        out = unnormalize(t)
        assert out.shape == (64, 64, 3), f"Expected (64, 64, 3), got {out.shape}"
        assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    def test_values_in_range(self):
        """Output values must lie in [0, 255]."""
        from visualization.visualize import unnormalize

        import torch

        t = torch.from_numpy(np.random.randn(3, 64, 64).astype(np.float32))
        out = unnormalize(t)
        assert out.min() >= 0, f"Min value {out.min()} < 0"
        assert out.max() <= 255, f"Max value {out.max()} > 255"

    def test_identity_after_norm(self):
        """A tensor that was normalised from a known uint8 image should round-trip."""
        from visualization.visualize import unnormalize

        import torch

        # Create a known RGB image [0, 255]
        original = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Normalise using the same constants as the dataset
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normed = ((original.astype(np.float32) / 255.0) - mean) / std
        tensor = torch.from_numpy(normed.transpose(2, 0, 1).copy()).float()
        recovered = unnormalize(tensor)
        # Allow small rounding differences
        diff = np.abs(original.astype(np.int32) - recovered.astype(np.int32))
        assert diff.mean() < 2.0, f"Mean pixel difference {diff.mean():.2f} > 2"


class TestColorizeLabel:
    """Tests for the Cityscapes label colourisation helper."""

    def test_output_shape(self, dummy_label):
        """[H, W] label → [H, W, 3] colour image."""
        from visualization.visualize import colorize_label

        rgb = colorize_label(dummy_label)
        assert rgb.shape == (*dummy_label.shape, 3), (
            f"Expected {(*dummy_label.shape, 3)}, got {rgb.shape}"
        )

    def test_output_dtype(self, dummy_label):
        """Output must be uint8."""
        from visualization.visualize import colorize_label

        rgb = colorize_label(dummy_label)
        assert rgb.dtype == np.uint8, f"Expected uint8, got {rgb.dtype}"

    def test_all_19_classes_have_nonzero_color(self, dummy_label):
        """Every class ID 0..18 maps to a non-black colour."""
        from visualization.visualize import colorize_label, CITYSCAPES_PALETTE

        # Force every class to appear at least once
        label = np.zeros((64, 64), dtype=np.uint8)
        for cls_id in range(19):
            label[cls_id, 0] = cls_id  # one pixel per class
        rgb = colorize_label(label)
        for cls_id, expected_color in CITYSCAPES_PALETTE.items():
            pixel = rgb[cls_id, 0]
            assert tuple(pixel) == expected_color, (
                f"Class {cls_id}: expected {expected_color}, got {tuple(pixel)}"
            )

    def test_unknown_class_pixels_are_black(self):
        """Labels outside 0..18 should appear as black (0, 0, 0)."""
        from visualization.visualize import colorize_label

        label = np.full((16, 16), 255, dtype=np.uint8)
        rgb = colorize_label(label)
        assert (rgb == 0).all(), "Unknown labels should produce black pixels"


class TestCreateTrustOverlay:
    """Tests for the trust mask overlay helper."""

    def test_output_shape(self, dummy_rgb, dummy_trust_mask):
        """[H, W, 3] image + [H, W] mask → [H, W, 3] overlay."""
        from visualization.visualize import create_trust_overlay

        overlay = create_trust_overlay(dummy_rgb, dummy_trust_mask)
        assert overlay.shape == dummy_rgb.shape, (
            f"Expected {dummy_rgb.shape}, got {overlay.shape}"
        )

    def test_output_dtype(self, dummy_rgb, dummy_trust_mask):
        from visualization.visualize import create_trust_overlay

        overlay = create_trust_overlay(dummy_rgb, dummy_trust_mask)
        assert overlay.dtype == np.uint8

    def test_trusted_pixels_unchanged(self, dummy_rgb):
        """Pixels with mask=1.0 should retain their original colour."""
        from visualization.visualize import create_trust_overlay

        mask = np.ones((64, 64), dtype=np.float32)
        overlay = create_trust_overlay(dummy_rgb, mask)
        np.testing.assert_array_equal(overlay, dummy_rgb)

    def test_rejected_pixels_greyed(self, dummy_rgb, dummy_trust_mask):
        """Pixels with mask=0.0 should be desaturated (greyed out)."""
        from visualization.visualize import create_trust_overlay

        overlay = create_trust_overlay(dummy_rgb, dummy_trust_mask)
        # Bottom half (rejected) should have R=G=B per pixel (grey)
        grey_region = overlay[32:, :]
        r, g, b = grey_region[:, :, 0], grey_region[:, :, 1], grey_region[:, :, 2]
        # Allow small tolerance for JPEG-level differences
        assert np.allclose(r, g, atol=3), "Rejected pixels must have R ≈ G"
        assert np.allclose(g, b, atol=3), "Rejected pixels must have G ≈ B"

    def test_rejected_not_all_black(self, dummy_rgb, dummy_trust_mask):
        """Greyscaled rejected regions should not be pure black (unless input was)."""
        from visualization.visualize import create_trust_overlay

        # Use a non-black image
        bright = np.full((64, 64, 3), 128, dtype=np.uint8)
        overlay = create_trust_overlay(bright, dummy_trust_mask)
        grey_region = overlay[32:, :]
        # The grey value should be the luminance ~= 128, not 0
        assert grey_region.mean() > 30, "Rejected region should not be near-black"


# ===========================================================================
# CLI argument parsing tests
# ===========================================================================


class TestCliArgumentParsing:
    """Tests for the CLI argument parser."""

    def test_minimal_args(self):
        """Minimal required args (checkpoint + config) must parse."""
        from visualization.visualize import _parse_args

        test_args = [
            "--checkpoint", "/path/to/last.pth",
            "--config", "configs/satg_hard.yaml",
        ]
        args = _parse_args(test_args)
        assert args.checkpoint == "/path/to/last.pth"
        assert args.config == "configs/satg_hard.yaml"
        assert args.num_images == 10, f"Expected default num_images=10, got {args.num_images}"
        assert args.output_dir == "visualizations/", (
            f"Expected default output_dir='visualizations/', got {args.output_dir!r}"
        )

    def test_all_args(self):
        """All CLI arguments must parse correctly."""
        from visualization.visualize import _parse_args

        test_args = [
            "--checkpoint", "ckpt.pth",
            "--config", "configs/satg_soft_weight.yaml",
            "--num_images", "5",
            "--output_dir", "my_vis/",
        ]
        args = _parse_args(test_args)
        assert args.checkpoint == "ckpt.pth"
        assert args.config == "configs/satg_soft_weight.yaml"
        assert args.num_images == 5
        assert args.output_dir == "my_vis/"

    def test_num_images_must_be_int(self):
        """num_images must be an integer (not a string)."""
        from visualization.visualize import _parse_args

        with pytest.raises(SystemExit):
            _parse_args([
                "--checkpoint", "ckpt.pth",
                "--config", "configs/satg_hard.yaml",
                "--num_images", "not_a_number",
            ])

    def test_missing_checkpoint_raises(self):
        """--checkpoint is required."""
        from visualization.visualize import _parse_args

        with pytest.raises(SystemExit):
            _parse_args([
                "--config", "configs/satg_hard.yaml",
            ])

    def test_missing_config_raises(self):
        """--config is required."""
        from visualization.visualize import _parse_args

        with pytest.raises(SystemExit):
            _parse_args([
                "--checkpoint", "ckpt.pth",
            ])


# ===========================================================================
# 5-panel figure generation tests
# ===========================================================================


class TestFivePanelFigure:
    """Tests for the 5-panel figure generation."""

    def test_five_panels_created(self, tmp_path: Path, dummy_rgb: np.ndarray,
                                 dummy_label: np.ndarray, dummy_confidence: np.ndarray,
                                 dummy_trust_mask: np.ndarray, dummy_weights: np.ndarray,
                                 vis_cfg: OmegaConf):
        """The generated figure must have exactly 5 subplots in a 1×5 layout."""
        from visualization.visualize import create_figure

        fig = create_figure(
            image_rgb=dummy_rgb,
            confidence=dummy_confidence,
            struct_heatmap=np.random.rand(64, 64).astype(np.float32),
            trust_mask=dummy_trust_mask,
            pseudo_label=dummy_label,
            gate_type="hard",
        )
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 5, f"Expected 5 subplots, got {len(axes)}"
        plt.close(fig)

    def test_panels_have_correct_titles(self, tmp_path: Path, dummy_rgb: np.ndarray,
                                        dummy_label: np.ndarray, dummy_confidence: np.ndarray,
                                        dummy_trust_mask: np.ndarray,
                                        vis_cfg: OmegaConf):
        """Each subplot must have its expected title."""
        from visualization.visualize import create_figure

        fig = create_figure(
            image_rgb=dummy_rgb,
            confidence=dummy_confidence,
            struct_heatmap=np.random.rand(64, 64).astype(np.float32),
            trust_mask=dummy_trust_mask,
            pseudo_label=dummy_label,
            gate_type="hard",
        )
        axes = fig.get_axes()
        expected_titles = [
            "Input Image",
            "Teacher Confidence",
            "Structural Heatmap",
            "Trust Mask (hard)",
            "SAT Pseudo-Label",
        ]
        for ax, expected in zip(axes, expected_titles):
            assert ax.get_title() == expected, (
                f"Expected title '{expected}', got '{ax.get_title()}'"
            )
        plt.close(fig)

    def test_figure_soft_weight_type_title(self, tmp_path: Path, dummy_rgb: np.ndarray,
                                           dummy_label: np.ndarray,
                                           dummy_confidence: np.ndarray,
                                           dummy_weights: np.ndarray,
                                           vis_cfg: OmegaConf):
        """Panel 4 title reflects the trust gate type."""
        from visualization.visualize import create_figure

        fig = create_figure(
            image_rgb=dummy_rgb,
            confidence=dummy_confidence,
            struct_heatmap=np.random.rand(64, 64).astype(np.float32),
            trust_mask=dummy_weights,
            pseudo_label=dummy_label,
            gate_type="soft_weight",
        )
        axes = fig.get_axes()
        assert "soft_weight" in axes[3].get_title()
        plt.close(fig)


# ===========================================================================
# End-to-end (mocked) pipeline tests
# ===========================================================================


class TestEndToEnd:
    """E2E test for the main() pipeline with all dependencies mocked."""

    @pytest.fixture(autouse=True)
    def _mock_segmentation_model(self, monkeypatch):
        """Mock SegmentationModel to avoid downloading pretrained weights."""
        import torch
        import torch.nn as nn

        class MockSegModel(nn.Module):
            """Lightweight stand-in for SegmentationModel."""

            def __init__(self, num_classes=19):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                B, _, H, W = x.shape
                main = torch.randn(B, 19, H // 8, W // 8)
                aux = torch.randn(B, 19, H // 8, W // 8)
                return main, aux

        monkeypatch.setattr(
            "visualization.visualize.SegmentationModel", MockSegModel
        )

    @pytest.fixture(autouse=True)
    def _mock_torch_load(self, monkeypatch):
        """Mock torch.load to return a minimal state dict (accepts **kwargs)."""
        import torch

        def mock_load(path, map_location=None, **kwargs):
            # Return a state dict with mostly zeros (just need it to be loadable)
            state = {}
            # The actual conv weights shape — just enough to not crash
            state["_model.classifier.0.weight"] = torch.zeros((256, 256, 3, 3))
            state["_model.classifier.0.bias"] = torch.zeros(256)
            state["_model.classifier.1.weight"] = torch.zeros(256, 256, 3, 3)
            state["_model.classifier.1.bias"] = torch.zeros(256)
            state["_model.classifier.2.weight"] = torch.zeros(256, 256, 3, 3)
            state["_model.classifier.2.bias"] = torch.zeros(256)
            state["_model.classifier.3.weight"] = torch.zeros(256, 256, 3, 3)
            state["_model.classifier.3.bias"] = torch.zeros(256)
            state["_model.classifier.4.weight"] = torch.zeros((19, 256, 1, 1))
            state["_model.classifier.4.bias"] = torch.zeros(19)
            state["_model.aux_classifier.0.weight"] = torch.zeros((256, 256, 3, 3))
            state["_model.aux_classifier.0.bias"] = torch.zeros(256)
            state["_model.aux_classifier.1.weight"] = torch.zeros(256, 256, 3, 3)
            state["_model.aux_classifier.1.bias"] = torch.zeros(256)
            state["_model.aux_classifier.2.weight"] = torch.zeros(256, 256, 3, 3)
            state["_model.aux_classifier.2.bias"] = torch.zeros(256)
            state["_model.aux_classifier.3.weight"] = torch.zeros(256, 256, 3, 3)
            state["_model.aux_classifier.3.bias"] = torch.zeros(256)
            state["_model.aux_classifier.4.weight"] = torch.zeros((19, 256, 1, 1))
            state["_model.aux_classifier.4.bias"] = torch.zeros(19)
            return {"model_state": state}

        monkeypatch.setattr("visualization.visualize.torch.load", mock_load)

    @pytest.fixture(autouse=True)
    def _mock_dataset(self, monkeypatch):
        """Mock CityscapesDataset to return a single dummy image-label pair."""

        class MockDataset:
            """Dummy dataset returning one synthetic sample (no batch dim).

            The DataLoader (also mocked) returns items as-is from __getitem__
            so we include a batch dim of 1 to simulate batch_size=1.
            """

            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                import torch
                img = torch.randn(1, 3, 64, 64).float()   # [B, C, H, W]
                label = torch.randint(0, 19, (1, 64, 64), dtype=torch.long)  # [B, H, W]
                return img, label

        monkeypatch.setattr(
            "visualization.visualize.CityscapesDataset", MockDataset
        )

    @pytest.fixture(autouse=True)
    def _mock_structural_prior(self, monkeypatch):
        """Mock StructuralPrior to return a dummy heatmap without OpenCV deps."""
        import numpy as np

        def mock_compute(self, image_rgb):
            H, W = image_rgb.shape[:2]
            return {
                "combined": np.random.rand(H, W).astype(np.float32),
                "edge": np.random.rand(H, W).astype(np.float32),
                "var": np.random.rand(H, W).astype(np.float32),
                "ent": np.random.rand(H, W).astype(np.float32),
                "corn": np.random.rand(H, W).astype(np.float32),
            }

        monkeypatch.setattr(
            "visualization.visualize.StructuralPrior.compute", mock_compute
        )

    @pytest.fixture(autouse=True)
    def _mock_dataloader(self, monkeypatch):
        """Mock DataLoader with a simple class (no torch inheritance)."""

        class MockDataLoader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
                self.batch_size = kwargs.get("batch_size", 1)

            def __len__(self):
                return len(self.dataset)

            def __iter__(self):
                class Iter:
                    def __init__(self, ds):
                        self._ds = ds
                        self._idx = 0

                    def __next__(self):
                        if self._idx >= len(self._ds):
                            raise StopIteration
                        self._idx += 1
                        return self._ds[self._idx - 1]

                return Iter(self.dataset)

        monkeypatch.setattr(
            "visualization.visualize.DataLoader", MockDataLoader
        )

    def test_e2e_creates_png_files(self, tmp_path: Path, vis_cfg: OmegaConf):
        """main() must create a .png file in the output directory."""
        from visualization.visualize import main

        out_dir = tmp_path / "vis_output"
        test_args = [
            "visualize.py",
            "--checkpoint", str(tmp_path / "dummy.pth"),
            "--config", "configs/satg_hard.yaml",
            "--num_images", "1",
            "--output_dir", str(out_dir),
        ]

        # Mock OmegaConf.load to return our vis_cfg for the config file
        def _mock_omega_load(path):
            if "default.yaml" in str(path):
                return OmegaConf.create({
                    "seed": 42,
                    "model": {"num_classes": 19, "backbone": "resnet50", "pretrained": True,
                              "backbone_lr_multiplier": 0.1},
                    "ema": {"momentum": 0.999, "warmup_iterations": 0},
                    "structural_prior": vis_cfg.structural_prior,
                    "trust_gate": vis_cfg.trust_gate,
                    "training": {"target_root": "./data/cityscapes", "crop_size": [512, 512],
                                 "num_workers": 2, "source_root": "./data/GTA5",
                                 "batch_size": 4, "iterations": 10,
                                 "eval_interval": 2000, "lr": 2.5e-4,
                                 "lambda_target": 1.0, "aux_loss_weight": 0.4,
                                 "rare_class_sampling": False,
                                 "poly_power": 0.9, "optimizer_momentum": 0.9,
                                 "weight_decay": 5e-4, "use_amp": False,
                                 "skip_heatmap": False, "label_suffix": "_trainids.png",
                                 "gradient_clip_norm": 5.0, "cudnn_benchmark": False},
                    "checkpoint": {"save_dir": "checkpoints/", "save_last_every": 2000},
                    "logging": {"backend": "wandb", "project": "satg-uda",
                                "log_every": 50, "csv_log": True},
                })
            # For the variant config, just override trust_gate type
            return OmegaConf.create({"trust_gate": vis_cfg.trust_gate})

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("visualization.visualize.OmegaConf.load", _mock_omega_load)
        monkeypatch.setattr("sys.argv", test_args)
        monkeypatch.setattr("visualization.visualize.Path", tmp_path.__class__)

        try:
            main()
        except Exception as e:
            pytest.fail(f"main() raised unexpected exception: {e}")

        # Check that PNG files were created
        expected_dir = out_dir / "satg_hard"
        png_files = list(expected_dir.glob("*.png"))
        assert len(png_files) > 0, (
            f"No PNG files found in {expected_dir}. "
            f"Contents: {list(expected_dir.iterdir()) if expected_dir.exists() else 'dir missing'}"
        )
        monkeypatch.undo()

    def test_e2e_uses_mocked_config_loading(self, tmp_path: Path, vis_cfg: OmegaConf):
        """The pipeline must use OmegaConf.merge to load and merge configs."""
        from visualization.visualize import main

        out_dir = tmp_path / "vis_out2"
        test_args = [
            "visualize.py",
            "--checkpoint", str(tmp_path / "dummy.pth"),
            "--config", "configs/satg_hard.yaml",
            "--num_images", "1",
            "--output_dir", str(out_dir),
        ]

        # Partial mock: return a minimal merged config
        merged = OmegaConf.merge(
            OmegaConf.create({
                "seed": 42,
                "model": {"num_classes": 19, "backbone": "resnet50", "pretrained": True,
                          "backbone_lr_multiplier": 0.1},
                "ema": {"momentum": 0.999, "warmup_iterations": 0},
                "structural_prior": vis_cfg.structural_prior,
                "trust_gate": vis_cfg.trust_gate,
                "training": {"target_root": "./data/cityscapes", "crop_size": [512, 512],
                             "num_workers": 2, "source_root": "./data/GTA5",
                             "batch_size": 4, "iterations": 10,
                             "eval_interval": 2000, "lr": 2.5e-4,
                             "lambda_target": 1.0, "aux_loss_weight": 0.4,
                             "rare_class_sampling": False,
                             "poly_power": 0.9, "optimizer_momentum": 0.9,
                             "weight_decay": 5e-4, "use_amp": False,
                             "skip_heatmap": False, "label_suffix": "_trainids.png",
                             "gradient_clip_norm": 5.0, "cudnn_benchmark": False},
                "checkpoint": {"save_dir": "checkpoints/", "save_last_every": 2000},
                "logging": {"backend": "wandb", "project": "satg-uda",
                            "log_every": 50, "csv_log": True},
            }),
            OmegaConf.create({"trust_gate": vis_cfg.trust_gate}),
        )

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("visualization.visualize.OmegaConf.load", lambda p: merged)
        monkeypatch.setattr("sys.argv", test_args)

        try:
            main()
        except Exception as e:
            pytest.fail(f"main() raised unexpected exception: {e}")

        expected_dir = out_dir / "satg_hard"
        png_files = list(expected_dir.glob("*.png"))
        assert len(png_files) > 0
        for f in png_files:
            assert f.suffix == ".png", f"Expected .png file, got {f.suffix}"
        monkeypatch.undo()
