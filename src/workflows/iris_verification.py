"""Iris verification workflow (with optional blink-based liveness)."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..capture.camera import Camera
from ..iris.recognition import IrisRecognitionSystem
from ..database.storage import DatabaseManager
from ..utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class IrisVerificationResult:
    success: bool
    user_id: Optional[str] = None
    left_distance: float = 999.0
    right_distance: float = 999.0
    confidence: float = 0.0
    message: str = ""
    liveness_passed: bool = False


class IrisVerificationWorkflow:

    def __init__(
        self,
        camera: Optional[Camera] = None,
        iris_system: Optional[IrisRecognitionSystem] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.config = get_config()
        self.camera = camera or Camera()
        self.iris_system = iris_system or IrisRecognitionSystem()
        self.db = db_manager or DatabaseManager()
        self.db.initialize()

        self.threshold = self.config.iris.threshold
        self.require_liveness = self.config.verification.require_liveness
        self.liveness = self.config.liveness

    def verify(self, user_id: str) -> IrisVerificationResult:
        left_template = self.db.get_iris_template(user_id, "left")
        right_template = self.db.get_iris_template(user_id, "right")

        if left_template is None or right_template is None:
            return IrisVerificationResult(
                success=False,
                user_id=user_id,
                message="Stored iris template not found",
            )

        if not self.camera.open():
            return IrisVerificationResult(
                success=False,
                message="Failed to open camera",
            )

        try:
            frame, liveness_passed, msg = self._capture_live_frame()

            if frame is None:
                return IrisVerificationResult(
                    success=False,
                    user_id=user_id,
                    liveness_passed=liveness_passed,
                    message=msg,
                )

            eyes = self.iris_system.extract_eye_regions(frame)
            if eyes is None:
                return IrisVerificationResult(
                    success=False,
                    user_id=user_id,
                    liveness_passed=liveness_passed,
                    message="Eyes not detected",
                )

            left_eye_img = self.iris_system.crop_eye(frame, eyes["left_eye"])
            right_eye_img = self.iris_system.crop_eye(frame, eyes["right_eye"])

            # Gate on crop quality before templating: an empty/edge crop would
            # otherwise crash generate_iris_template (cvtColor on a 0-size array).
            if not (self.iris_system.is_acceptable(left_eye_img)
                    and self.iris_system.is_acceptable(right_eye_img)):
                return IrisVerificationResult(
                    success=False,
                    user_id=user_id,
                    liveness_passed=liveness_passed,
                    message="Eye image quality too low",
                )

            live_left = self.iris_system.generate_iris_template(left_eye_img)
            live_right = self.iris_system.generate_iris_template(right_eye_img)

            left_distance = self.iris_system.compare_iris_templates(live_left, left_template)
            right_distance = self.iris_system.compare_iris_templates(live_right, right_template)

            avg_distance = (left_distance + right_distance) / 2

            logger.debug("iris threshold=%.3f avg_distance=%.3f", self.threshold, avg_distance)

            success = avg_distance < self.threshold
            confidence = max(0.0, (1 - avg_distance / max(self.threshold, 1e-6)) * 100)

            return IrisVerificationResult(
                success=success,
                user_id=user_id,
                left_distance=left_distance,
                right_distance=right_distance,
                confidence=confidence,
                liveness_passed=liveness_passed,
                message="Iris Verified" if success else "Iris Verification Failed",
            )

        finally:
            self.camera.close()
            cv2.destroyAllWindows()

    def _capture_live_frame(self):
        """Capture a frame for matching, enforcing blink liveness if enabled.

        Returns ``(frame, liveness_passed, message)``. When liveness is required,
        the user must blink (EAR dips below threshold then recovers) within the
        timeout; a still photo of a face will not blink and is rejected.
        """
        if not self.require_liveness:
            capture = self.camera.capture_with_preview(
                window_name="Iris Verification",
                instruction="Align eyes and press SPACE to verify",
            )
            if not capture.success or capture.frame is None:
                return None, False, "Failed to capture frame"
            return capture.frame, False, "Liveness disabled"

        window = "Iris Verification (blink to confirm liveness)"
        ear_thresh = self.liveness.ear_threshold
        open_thresh = ear_thresh * 1.1  # hysteresis to avoid double-counting
        deadline = time.time() + self.liveness.timeout_seconds

        blinks = 0
        eye_closed = False
        good_frame = None

        while time.time() < deadline:
            capture = self.camera.read_frame()
            if not capture.success or capture.frame is None:
                continue

            frame = capture.frame
            eyes = self.iris_system.extract_eye_regions(frame)

            status = "No eyes detected"
            color = (0, 0, 255)

            if eyes is not None:
                ear = self.iris_system.average_ear(eyes)

                # Blink state machine with hysteresis.
                if ear < ear_thresh:
                    eye_closed = True
                elif eye_closed and ear >= open_thresh:
                    eye_closed = False
                    blinks += 1

                # Keep the most recent open-eyed, good-quality frame for matching.
                if ear >= open_thresh:
                    left_crop = self.iris_system.crop_eye(frame, eyes["left_eye"])
                    right_crop = self.iris_system.crop_eye(frame, eyes["right_eye"])
                    if self.iris_system.is_acceptable(left_crop) and self.iris_system.is_acceptable(right_crop):
                        good_frame = frame.copy()

                status = f"Blinks: {blinks}/{self.liveness.min_blinks}  EAR: {ear:.2f}"
                color = (0, 255, 0) if blinks >= self.liveness.min_blinks else (0, 165, 255)

            preview = frame.copy()
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(preview, "Blink naturally to verify (Q to cancel)",
                        (10, preview.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                cv2.destroyWindow(window)
                return None, False, "Cancelled by user"

            if blinks >= self.liveness.min_blinks and good_frame is not None:
                cv2.destroyWindow(window)
                return good_frame, True, "Liveness confirmed"

        cv2.destroyWindow(window)
        if blinks >= self.liveness.min_blinks:
            # Blinked but never got a clean frame; fall back to last attempt.
            return good_frame, True, "Liveness confirmed"
        return None, False, "Liveness check failed (no blink detected)"
