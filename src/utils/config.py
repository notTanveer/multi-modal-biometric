"""
Configuration management for the biometric system.
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field


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


class VerificationConfig(BaseModel):
    max_attempts: int = 3
    timeout_seconds: int = 30
    require_liveness: bool = False


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
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            return Config(**yaml_config)
    else:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return Config()


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
