"""Unit tests for the pure score-level fusion helpers (no hardware)."""

import pytest

from src.workflows.fusion import combine_scores, normalize


@pytest.mark.parametrize(
    "conf, expected",
    [
        (0.0, 0.0),
        (50.0, 0.5),
        (100.0, 1.0),
        (-10.0, 0.0),    # clamped low
        (150.0, 1.0),    # clamped high
    ],
)
def test_normalize_clamps_to_unit_interval(conf, expected):
    assert normalize(conf) == pytest.approx(expected)


def test_combine_scores_is_weighted_sum():
    # 0.6 * 1.0 + 0.4 * 0.0 = 0.6
    assert combine_scores(1.0, 0.0, 0.6, 0.4) == pytest.approx(0.6)
    # 0.6 * 0.0 + 0.4 * 1.0 = 0.4
    assert combine_scores(0.0, 1.0, 0.6, 0.4) == pytest.approx(0.4)


def test_combine_scores_bounds():
    # Both factors maxed with weights summing to 1.0 -> exactly 1.0
    assert combine_scores(1.0, 1.0, 0.6, 0.4) == pytest.approx(1.0)
    # Both zero -> 0.0
    assert combine_scores(0.0, 0.0, 0.6, 0.4) == pytest.approx(0.0)


def test_combine_scores_matches_manual_formula():
    f_norm, i_norm, f_w, i_w = 0.82, 0.71, 0.6, 0.4
    assert combine_scores(f_norm, i_norm, f_w, i_w) == pytest.approx(
        f_w * f_norm + i_w * i_norm
    )
