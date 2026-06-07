"""
Configuration management for the biometric system.
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class CameraConfig(BaseModel):
    device_id: int = 0
    resolution: dict = Field(default_factory=lambda: {"width": 640, "height": 480})
    fps: int = 30
    warmup_frames: int = 10


class FaceRecognitionConfig(BaseModel):
    detection_model: str = "hog"
    encoding_model: str = "small"
    num_jitters: int = 1
    match_tolerance: float = 0.6
    strict_tolerance: float = 0.5
    min_face_size: int = 50

    @field_validator("detection_model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        if v not in {"hog", "cnn"}:
            raise ValueError(f"detection_model must be 'hog' or 'cnn', got '{v}'")
        return v

    @field_validator("match_tolerance", "strict_tolerance")
    @classmethod
    def _check_tolerance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tolerance must be > 0, got {v}")
        return v


class QualityConfig(BaseModel):
    min_brightness: int = 40
    max_brightness: int = 220
    min_sharpness: float = 50
    face_detection_confidence: float = 0.8


class DatabaseConfig(BaseModel):
    path: str = "data/biometric.db"
    encrypt_templates: bool = False


class EnrollmentConfig(BaseModel):
    num_samples: int = 5
    sample_interval: float = 0.5
    require_quality_check: bool = True


class LivenessConfig(BaseModel):
    """Blink / eye-aspect-ratio anti-spoof settings."""
    ear_threshold: float = 0.21       # Eye-aspect-ratio below this = eye closed
    min_blinks: int = 1               # Blinks required to consider input "live"
    timeout_seconds: float = 6.0      # Max time to wait for a blink

    @field_validator("ear_threshold")
    @classmethod
    def _check_ear(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"ear_threshold must be in (0, 1), got {v}")
        return v

    @field_validator("min_blinks")
    @classmethod
    def _check_min_blinks(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"min_blinks must be >= 0, got {v}")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def _check_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {v}")
        return v


class VerificationConfig(BaseModel):
    max_attempts: int = 3
    timeout_seconds: int = 30
    require_liveness: bool = True


class IrisConfig(BaseModel):
    """Iris (eye-region appearance) recognition settings."""
    threshold: float = 0.65           # Distance threshold (lower = stricter)
    num_samples: int = 3              # Eye samples to average at enrollment
    min_sharpness: float = 15.0       # Laplacian-variance gate for eye crops
    metric: str = "rmse"              # "rmse" or "correlation"

    @field_validator("metric")
    @classmethod
    def _check_metric(cls, v: str) -> str:
        if v not in {"rmse", "correlation"}:
            raise ValueError(f"metric must be 'rmse' or 'correlation', got '{v}'")
        return v

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"threshold must be > 0, got {v}")
        return v

    @field_validator("num_samples")
    @classmethod
    def _check_num_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_samples must be >= 1, got {v}")
        return v


class FusionConfig(BaseModel):
    """Score-level fusion settings for multi-modal decision."""
    face_weight: float = 0.6
    iris_weight: float = 0.4
    combined_threshold: float = 0.6   # On the normalized [0,1] weighted score

    @field_validator("face_weight", "iris_weight")
    @classmethod
    def _check_weight(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"fusion weights must be in [0, 1], got {v}")
        return v

    @field_validator("combined_threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"combined_threshold must be in (0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _check_weights_sum(self) -> "FusionConfig":
        total = self.face_weight + self.iris_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"face_weight + iris_weight must sum to 1.0, got {total:.4f}"
            )
        return self


class SystemConfig(BaseModel):
    name: str = "Multi-Modal Biometric Auth"
    version: str = "0.1.0"
    debug: bool = True


class Config(BaseModel):
    """Main configuration class."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    face_recognition: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    enrollment: EnrollmentConfig = Field(default_factory=EnrollmentConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    iris: IrisConfig = Field(default_factory=IrisConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    liveness: LivenessConfig = Field(default_factory=LivenessConfig)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Config object with all settings.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "settings.yaml"
    
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return Config()

    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in config at {config_path}: {exc}") from exc

    try:
        return Config(**yaml_config)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid config at {config_path}: {exc}") from exc


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config
