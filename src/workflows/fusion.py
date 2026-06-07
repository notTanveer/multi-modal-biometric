"""Pure score-level fusion helpers.

These are the deterministic arithmetic pieces of the multi-modal decision,
factored out so the GUI (``app.py``), the in-process facade
(``BiometricService.fuse``) and ``MultiModalVerificationWorkflow`` all share one
implementation instead of three hand-copied formulas. The module deliberately
imports nothing heavy (no OpenCV / dlib / camera), so it is trivially unit-testable
without hardware.
"""


def normalize(confidence: float) -> float:
    """Map a 0-100 per-factor confidence onto a clamped [0, 1] fusion score."""
    return max(0.0, min(1.0, confidence / 100.0))


def combine_scores(
    face_norm: float,
    iris_norm: float,
    face_weight: float,
    iris_weight: float,
) -> float:
    """Weighted sum of the two normalized [0, 1] factor scores.

    The result stays in [0, 1] as long as the weights sum to 1.0 and each input
    is already normalized; both invariants are enforced upstream (config
    validation + :func:`normalize`).
    """
    return face_weight * face_norm + iris_weight * iris_norm
