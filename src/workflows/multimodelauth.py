import time
from dataclasses import dataclass
from typing import Optional

from .verification import FaceVerificationWorkflow
from .iris_verification import IrisVerificationWorkflow


@dataclass
class MultiModalVerificationResult:
    success: bool
    user_id: Optional[str] = None

    face_success: bool = False
    iris_success: bool = False

    face_confidence: float = 0.0
    iris_confidence: float = 0.0

    combined_confidence: float = 0.0
    message: str = ""


class MultiModalVerificationWorkflow:

    def __init__(self):
        self.face_workflow = FaceVerificationWorkflow()
        self.iris_workflow = IrisVerificationWorkflow()

        # Weighting for score fusion
        self.face_weight = 0.6
        self.iris_weight = 0.4

        self.combined_threshold = 70.0

    def verify(self, user_id: str) -> MultiModalVerificationResult:
        """
        Perform 2-step verification:
        1. Face Verification
        2. Iris Verification
        """

        print("\n" + "="*60)
        print("STEP 1: FACE VERIFICATION")
        print("="*60)

        face_result = self.face_workflow.verify(user_id)

        if not face_result.success:
            return MultiModalVerificationResult(
                success=False,
                user_id=user_id,
                face_success=False,
                iris_success=False,
                face_confidence=face_result.confidence,
                iris_confidence=0.0,
                combined_confidence=0.0,
                message="Face verification failed"
            )

        print("\n" + "="*60)
        print("STEP 2: IRIS VERIFICATION")
        print("="*60)

        iris_result = self.iris_workflow.verify(user_id)

        if not iris_result.success:
            return MultiModalVerificationResult(
                success=False,
                user_id=user_id,
                face_success=True,
                iris_success=False,
                face_confidence=face_result.confidence,
                iris_confidence=iris_result.confidence,
                combined_confidence=0.0,
                message="Iris verification failed"
            )

        combined_confidence = (
                                      face_result.confidence + iris_result.confidence
                              ) / 2

        success = face_result.success and iris_result.success

        return MultiModalVerificationResult(
            success=success,
            user_id=user_id,
            face_success=True,
            iris_success=True,
            face_confidence=face_result.confidence,
            iris_confidence=iris_result.confidence,
            combined_confidence=combined_confidence,
            message="Authentication Successful" if success else "Verification Failed"
        )