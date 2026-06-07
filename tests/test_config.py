"""Unit tests for configuration loading and validation."""

import textwrap

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import Config, FusionConfig, IrisConfig, load_config


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.fusion.face_weight + cfg.fusion.iris_weight == pytest.approx(1.0)
    assert cfg.iris.metric in {"rmse", "correlation"}


def test_fusion_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        FusionConfig(face_weight=0.9, iris_weight=0.4)


def test_fusion_weight_out_of_range_rejected():
    with pytest.raises(ValidationError):
        FusionConfig(face_weight=1.5, iris_weight=-0.5)


def test_iris_metric_must_be_known():
    with pytest.raises(ValidationError):
        IrisConfig(metric="bogus")


def test_iris_threshold_must_be_positive():
    with pytest.raises(ValidationError):
        IrisConfig(threshold=0.0)


def test_load_config_from_valid_yaml(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(textwrap.dedent("""
        fusion:
          face_weight: 0.7
          iris_weight: 0.3
          combined_threshold: 0.55
    """))
    cfg = load_config(str(path))
    assert cfg.fusion.face_weight == pytest.approx(0.7)
    assert cfg.fusion.combined_threshold == pytest.approx(0.55)


def test_load_config_wraps_validation_error(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("iris:\n  metric: nonsense\n")
    with pytest.raises(RuntimeError) as exc:
        load_config(str(path))
    assert str(path) in str(exc.value)


def test_load_config_wraps_yaml_error(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("fusion: [unbalanced\n")  # malformed YAML
    with pytest.raises(RuntimeError):
        load_config(str(path))


def test_load_config_missing_file_uses_defaults(tmp_path):
    cfg = load_config(str(tmp_path / "does_not_exist.yaml"))
    assert isinstance(cfg, Config)
