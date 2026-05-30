# src/workflows/iris_verification.py

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..capture.camera import Camera
from ..iris.recognition import IrisRecognitionSystem
from ..database.storage import DatabaseManager


@dataclass
class IrisVerificationResult:
    success: bool
    user_id: Optional[str] = None
    left_distance: float = 999.0
    right_distance: float = 999.0
    confidence: float = 0.0
    message: str = ""


class IrisVerificationWorkflow:

    def __init__(self):
        self.camera = Camera()
        self.iris_system = IrisRecognitionSystem()
        self.db = DatabaseManager()
        self.db.initialize()

        self.threshold = 0.65  # tune later

    def verify(self, user_id: str) -> IrisVerificationResult:

        left_template = self.db.get_iris_template(user_id, "left")
        right_template = self.db.get_iris_template(user_id, "right")

        if left_template is None or right_template is None:
            return IrisVerificationResult(
                success=False,
                user_id=user_id,
                message="Stored iris template not found"
            )

        if not self.camera.open():
            return IrisVerificationResult(
                success=False,
                message="Failed to open camera"
            )

        try:
            capture = self.camera.capture_with_preview(
                window_name="Iris Verification",
                instruction="Align eyes and press SPACE to verify"
            )

            if not capture.success or capture.frame is None:
                return IrisVerificationResult(
                    success=False,
                    message="Failed to capture frame"
                )

            frame = capture.frame

            eyes = self.iris_system.extract_eye_regions(frame)

            if eyes is None:
                return IrisVerificationResult(
                    success=False,
                    message="Eyes not detected"
                )

            left_eye_img = self.iris_system.crop_eye(frame, eyes["left_eye"])
            right_eye_img = self.iris_system.crop_eye(frame, eyes["right_eye"])



            live_left = self.iris_system.generate_iris_template(left_eye_img)
            live_right = self.iris_system.generate_iris_template(right_eye_img)

            left_distance = self.iris_system.compare_iris_templates(
                live_left,
                left_template
            )

            right_distance = self.iris_system.compare_iris_templates(
                live_right,
                right_template
            )

            avg_distance = (left_distance + right_distance) / 2

            print("IRIS THRESHOLD =", self.threshold)
            print("AVG DISTANCE =", avg_distance)

            success = avg_distance < self.threshold

            confidence = max(0, (1 - avg_distance / self.threshold) * 100)

            return IrisVerificationResult(
                success=success,
                user_id=user_id,
                left_distance=left_distance,
                right_distance=right_distance,
                confidence=confidence,
                message="Iris Verified" if success else "Iris Verification Failed"
            )

        finally:
            self.camera.close()