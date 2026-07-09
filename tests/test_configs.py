"""Config isolation verification tests.

Ensures each variant config:
- Loads without error via OmegaConf
- Merges correctly with default.yaml
- Has expected overrides
- Shares identical backbone / optimizer / source data keys
- Differs only in the target loss mechanism
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _load_merged(name: str, config_dir: Path) -> OmegaConf:
    """Load default.yaml then merge variant on top."""
    default = OmegaConf.load(str(config_dir / "default.yaml"))
    variant = OmegaConf.load(str(config_dir / f"{name}.yaml"))
    return OmegaConf.merge(default, variant)


# All variant configs + soft_label
VARIANT_NAMES = [
    "satg_hard",
    "satg_soft_weight",
    "satg_soft_label",
    "baseline_mean_teacher",
    "source_only",
]

# Keys that MUST be identical across ALL variants after merge
IDENTICAL_KEYS = [
    "model.backbone",
    "model.num_classes",
    "training.lr",
    "training.poly_power",
    "training.iterations",
    "training.batch_size",
    "training.crop_size",
    "ema.momentum",
    "seed",
]


class TestConfigLoading:
    """Every config file must load without error."""

    @pytest.mark.parametrize("name", VARIANT_NAMES)
    def test_variant_loads_without_error(self, config_dir: Path, name: str):
        path = config_dir / f"{name}.yaml"
        assert path.exists(), f"Config not found: {path}"
        cfg = OmegaConf.load(str(path))
        assert cfg is not None

    def test_default_loads_without_error(self, config_dir: Path):
        cfg = OmegaConf.load(str(config_dir / "default.yaml"))
        assert cfg is not None

    @pytest.mark.parametrize("name", VARIANT_NAMES)
    def test_merge_succeeds(self, config_dir: Path, name: str):
        cfg = _load_merged(name, config_dir)
        assert cfg is not None
        # After merge, all expected top-level keys must exist
        assert hasattr(cfg, "training")
        assert hasattr(cfg, "trust_gate")
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "structural_prior")


class TestConfigValues:
    """Specific value assertions per config (after merge)."""

    def test_source_only_lambda_target_zero(self, config_dir: Path):
        cfg = _load_merged("source_only", config_dir)
        assert (
            cfg.training.lambda_target == 0.0
        ), f"Expected lambda_target=0.0, got {cfg.training.lambda_target}"

    def test_mean_teacher_tau_struct_disabled(self, config_dir: Path):
        """Mean Teacher uses tau_struct=1.01 so H in [0,1] always passes structural gate."""
        cfg = _load_merged("baseline_mean_teacher", config_dir)
        assert (
            cfg.trust_gate.tau_struct == 1.01
        ), f"Expected tau_struct=1.01, got {cfg.trust_gate.tau_struct}"
        assert (
            cfg.training.skip_heatmap is True
        ), "Expected skip_heatmap=True for mean_teacher baseline"

    def test_soft_label_type(self, config_dir: Path):
        cfg = _load_merged("satg_soft_label", config_dir)
        assert (
            cfg.trust_gate.type == "soft_label"
        ), f"Expected type=soft_label, got {cfg.trust_gate.type}"

    def test_soft_label_params(self, config_dir: Path):
        cfg = _load_merged("satg_soft_label", config_dir)
        assert cfg.trust_gate.soft_label_k == 4.0
        assert cfg.trust_gate.soft_label_t_max == 5.0

    def test_hard_type(self, config_dir: Path):
        cfg = _load_merged("satg_hard", config_dir)
        assert cfg.trust_gate.type == "hard"

    def test_soft_weight_type(self, config_dir: Path):
        cfg = _load_merged("satg_soft_weight", config_dir)
        assert cfg.trust_gate.type == "soft_weight"


class TestConfigIsolation:
    """All variants share identical backbone/optimizer/source keys."""

    @pytest.mark.parametrize("key", IDENTICAL_KEYS)
    def test_shared_key_identical_across_variants(self, config_dir: Path, key: str):
        """For each shared key, all variants must have the same merged value."""
        values = []
        for name in VARIANT_NAMES:
            cfg = _load_merged(name, config_dir)
            try:
                val = OmegaConf.select(cfg, key)
            except Exception:
                val = None
            values.append((name, val))

        non_none = [(n, v) for n, v in values if v is not None]
        if len(non_none) < 2:
            return
        ref_name, ref_val = non_none[0]
        for name, val in non_none[1:]:
            assert val == ref_val, f"Key '{key}' differs: {ref_name}={ref_val}, {name}={val}"

    def test_target_loss_mechanism_differs(self, config_dir: Path):
        """Each variant must have a distinguishable target loss mechanism."""
        cfg_hard = _load_merged("satg_hard", config_dir)
        cfg_soft_weight = _load_merged("satg_soft_weight", config_dir)
        cfg_soft_label = _load_merged("satg_soft_label", config_dir)
        cfg_mt = _load_merged("baseline_mean_teacher", config_dir)
        cfg_so = _load_merged("source_only", config_dir)

        types = {
            "satg_hard": cfg_hard.trust_gate.type,
            "satg_soft_weight": cfg_soft_weight.trust_gate.type,
            "satg_soft_label": cfg_soft_label.trust_gate.type,
            "baseline_mean_teacher": cfg_mt.trust_gate.type,
        }
        # At least 3 distinct gate types (hard appears twice intentionally)
        gate_types = list(types.values())
        assert len(set(gate_types)) >= 3, f"Not enough distinct gate types: {types}"
        # Source only is unique via lambda_target=0.0
        assert cfg_so.training.lambda_target == 0.0
        # All others have non-zero lambda_target
        for name in ["satg_hard", "satg_soft_weight", "satg_soft_label", "baseline_mean_teacher"]:
            c = _load_merged(name, config_dir)
            assert (
                c.training.lambda_target != 0.0
            ), f"{name} has lambda_target=0.0 (should be source-only)"
