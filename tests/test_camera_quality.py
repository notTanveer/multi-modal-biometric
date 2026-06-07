"""Unit tests for Camera.assess_quality (no device is opened).

``Camera()`` only reads config in __init__; ``assess_quality`` is pure image math
on a supplied frame, so these run without a webcam.
"""

import numpy as np
import pytest

from src.capture.camera import Camera


@pytest.fixture(scope="module")
def camera():
    return Camera()


def test_dark_frame_flagged_too_dark(camera):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    q = camera.assess_quality(frame)
    assert not q.is_acceptable
    assert any("dark" in issue.lower() for issue in q.issues)


def test_bright_frame_flagged_too_bright(camera):
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)
    q = camera.assess_quality(frame)
    assert not q.is_acceptable
    assert any("bright" in issue.lower() for issue in q.issues)


def test_uniform_midgray_is_blurry(camera):
    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    q = camera.assess_quality(frame)
    # Brightness is fine but a flat image has ~zero Laplacian variance.
    assert q.sharpness < camera.config.quality.min_sharpness
    assert not q.is_acceptable


def test_textured_well_lit_frame_is_acceptable(camera):
    rng = np.random.default_rng(0)
    # Mean ~128 keeps brightness in range; noise gives high Laplacian variance.
    frame = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    q = camera.assess_quality(frame)
    assert q.is_acceptable
    assert q.issues == []
