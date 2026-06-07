"""Iris (eye-region appearance) recognition.

This is an *appearance-based* eye-region matcher built on RGB-webcam landmarks,
not a true NIR iris-code system, so accuracy is inherently limited. The code
here is hardened for robustness: symmetric padded crops, CLAHE lighting
normalization, a sharpness quality gate, multi-sample averaging support, and a
configurable distance metric. Eye-aspect-ratio (EAR) helpers support blink-based
liveness in the verification workflow.
"""

import logging

import cv2
import numpy as np
import face_recognition
from dataclasses import dataclass

from ..utils.config import get_config

logger = logging.getLogger(__name__)

TEMPLATE_SIZE = 64  # Output template is TEMPLATE_SIZE x TEMPLATE_SIZE, flattened


@dataclass
class IrisTemplate:
    eye: str
    features: np.ndarray
    quality_score: float


class IrisRecognitionSystem:

    def __init__(self):
        cfg = get_config().iris
        self.threshold = cfg.threshold
        self.min_sharpness = cfg.min_sharpness
        self.metric = cfg.metric
        # CLAHE gives far better lighting robustness than global equalizeHist.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Eye localization
    # ------------------------------------------------------------------
    def extract_eye_regions(self, frame: np.ndarray):
        """Return left/right eye landmark point arrays, or None if no face."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_list = face_recognition.face_landmarks(rgb)

        if not landmarks_list:
            return None

        landmarks = landmarks_list[0]
        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return None

        return {
            "left_eye": np.array(landmarks["left_eye"]),
            "right_eye": np.array(landmarks["right_eye"]),
        }

    def crop_eye(self, frame: np.ndarray, eye_points: np.ndarray, padding: int = 8):
        """Crop a padded box around the eye, clamped to frame bounds.

        The previous implementation subtracted padding from the origin but only
        added it to width/height, producing an asymmetric, off-centre crop. This
        version pads symmetrically and clamps to the image.
        """
        h, w = frame.shape[:2]
        x, y, bw, bh = cv2.boundingRect(eye_points)

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)

        return frame[y0:y1, x0:x1]

    # ------------------------------------------------------------------
    # Templating + quality
    # ------------------------------------------------------------------
    def assess_quality(self, eye_image: np.ndarray) -> float:
        """Sharpness score (Laplacian variance) of an eye crop."""
        if eye_image is None or eye_image.size == 0:
            return 0.0
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def is_acceptable(self, eye_image: np.ndarray) -> bool:
        """Reject crops that are empty or too blurry to be useful."""
        if eye_image is None or eye_image.size == 0:
            return False
        if min(eye_image.shape[:2]) < 6:
            return False
        return self.assess_quality(eye_image) >= self.min_sharpness

    def generate_iris_template(self, eye_image: np.ndarray) -> np.ndarray:
        """Produce a normalized, flattened feature vector for an eye crop."""
        if eye_image is None or eye_image.size == 0 or min(eye_image.shape[:2]) < 1:
            raise ValueError(
                "Cannot generate an iris template from an empty eye crop; "
                "gate the crop with is_acceptable() first."
            )
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (TEMPLATE_SIZE, TEMPLATE_SIZE))
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = self._clahe.apply(gray)

        return gray.flatten().astype("float32") / 255.0

    @staticmethod
    def average_templates(templates: list[np.ndarray]) -> np.ndarray:
        """Average several per-eye templates into one robust template."""
        return np.mean(np.stack(templates, axis=0), axis=0).astype("float32")

    def compare_iris_templates(self, template1: np.ndarray, template2: np.ndarray) -> float:
        """Distance between two templates (lower = more similar).

        ``rmse`` is the root-mean-square pixel difference. ``correlation`` uses
        ``1 - normalized_cross_correlation`` so it is more tolerant of uniform
        brightness shifts. Both return a value where ``< threshold`` means match.

        A shape mismatch means the stored template predates the current
        (CLAHE / symmetric-crop / averaged) pipeline and is incompatible — return
        ``inf`` so it reads as a clean non-match (prompting a re-enroll) instead
        of raising a NumPy broadcasting error.
        """
        if template1.shape != template2.shape:
            logger.warning(
                "Iris template shape mismatch (%s vs %s); stored template is "
                "incompatible with the current matcher — re-enroll this user.",
                template1.shape, template2.shape,
            )
            return float("inf")

        if self.metric == "correlation":
            a = template1 - template1.mean()
            b = template2 - template2.mean()
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            ncc = float(np.dot(a, b) / denom)
            return 1.0 - ncc

        return float(((template1 - template2) ** 2).mean() ** 0.5)

    # ------------------------------------------------------------------
    # Liveness (eye-aspect-ratio / blink)
    # ------------------------------------------------------------------
    @staticmethod
    def eye_aspect_ratio(eye_points: np.ndarray) -> float:
        """Eye-aspect-ratio (EAR) from 6 eye landmark points.

        EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|). Drops toward 0 when the eye
        closes, so a dip-then-recover indicates a blink.
        """
        if eye_points is None or len(eye_points) < 6:
            return 1.0
        p = eye_points.astype("float64")
        vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        horizontal = 2.0 * np.linalg.norm(p[0] - p[3]) + 1e-8
        return float(vertical / horizontal)

    def average_ear(self, eyes: dict) -> float:
        """Mean EAR across both eyes for a frame's eye regions."""
        return 0.5 * (
            self.eye_aspect_ratio(eyes["left_eye"])
            + self.eye_aspect_ratio(eyes["right_eye"])
        )
