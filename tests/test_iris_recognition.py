"""Unit tests for the deterministic parts of the iris matcher (no camera).

These exercise the math and the robustness guards added in the hardening pass:
empty-crop rejection, template shape, and the shape-mismatch -> inf guard.
"""

import math

import numpy as np
import pytest

from src.iris.recognition import IrisRecognitionSystem, TEMPLATE_SIZE


@pytest.fixture(scope="module")
def iris():
    return IrisRecognitionSystem()


def _eye_points(open_amount: float) -> np.ndarray:
    """Six EAR landmark points for a synthetic eye.

    Horizontal corners are fixed; ``open_amount`` scales the vertical gap, so a
    small value mimics a (nearly) closed eye and a large value an open one.
    """
    return np.array(
        [
            [0, 5],                       # p0 left corner
            [3, 5 - open_amount],         # p1 top
            [7, 5 - open_amount],         # p2 top
            [10, 5],                      # p3 right corner
            [7, 5 + open_amount],         # p4 bottom
            [3, 5 + open_amount],         # p5 bottom
        ],
        dtype=np.float64,
    )


def test_eye_aspect_ratio_higher_when_open(iris):
    open_ear = iris.eye_aspect_ratio(_eye_points(open_amount=4.0))
    closed_ear = iris.eye_aspect_ratio(_eye_points(open_amount=0.2))
    assert open_ear > closed_ear
    assert closed_ear >= 0.0


def test_eye_aspect_ratio_degenerate_input(iris):
    # Fewer than 6 points -> safe default of 1.0 (treated as "open").
    assert iris.eye_aspect_ratio(np.zeros((3, 2))) == 1.0
    assert iris.eye_aspect_ratio(None) == 1.0


def test_generate_iris_template_shape_and_dtype(iris):
    crop = np.full((40, 30, 3), 120, dtype=np.uint8)
    template = iris.generate_iris_template(crop)
    assert template.shape == (TEMPLATE_SIZE * TEMPLATE_SIZE,)
    assert template.dtype == np.float32
    assert template.min() >= 0.0 and template.max() <= 1.0


def test_generate_iris_template_rejects_empty(iris):
    with pytest.raises(ValueError):
        iris.generate_iris_template(np.zeros((0, 0, 3), dtype=np.uint8))


def test_is_acceptable_rejects_empty_and_tiny(iris):
    assert iris.is_acceptable(None) is False
    assert iris.is_acceptable(np.zeros((0, 0, 3), dtype=np.uint8)) is False
    assert iris.is_acceptable(np.zeros((3, 3, 3), dtype=np.uint8)) is False  # < 6px


def test_average_templates_is_elementwise_mean(iris):
    a = np.zeros(TEMPLATE_SIZE * TEMPLATE_SIZE, dtype=np.float32)
    b = np.ones(TEMPLATE_SIZE * TEMPLATE_SIZE, dtype=np.float32)
    avg = iris.average_templates([a, b])
    assert np.allclose(avg, 0.5)


def test_compare_identical_templates_is_zero_distance(iris):
    t = np.linspace(0, 1, TEMPLATE_SIZE * TEMPLATE_SIZE).astype(np.float32)
    assert iris.compare_iris_templates(t, t) == pytest.approx(0.0, abs=1e-6)


def test_compare_rmse_increases_with_difference(iris):
    base = np.zeros(16, dtype=np.float32)
    near = np.full(16, 0.1, dtype=np.float32)
    far = np.full(16, 0.5, dtype=np.float32)
    iris.metric = "rmse"
    assert iris.compare_iris_templates(base, near) < iris.compare_iris_templates(base, far)


def test_compare_shape_mismatch_returns_inf(iris):
    a = np.zeros(TEMPLATE_SIZE * TEMPLATE_SIZE, dtype=np.float32)
    b = np.zeros(128, dtype=np.float32)
    assert math.isinf(iris.compare_iris_templates(a, b))
