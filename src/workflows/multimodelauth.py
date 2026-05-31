"""Multi-modal verification workflow.

Combines face and iris factors with score-level fusion. The decision policy is
**weighted-score fusion + AND gate**:

    combined = face_weight * face_norm + iris_weight * iris_norm
    success  = (combined >= combined_threshold) AND face_ok AND iris_ok

Both per-factor normalized scores are on a comparable [0, 1] scale (each factor's
confidence / 100), so the weighted sum is meaningful. All weights and the
threshold come from ``config/settings.yaml`` (``fusion:`` section).
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .verification import FaceVerificationWorkflow
from .iris_verification import IrisVerificationWorkflow
from ..database.storage import DatabaseManager
from ..utils.config import get_config

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        face_workflow: Optional[FaceVerificationWorkflow] = None,
        iris_workflow: Optional[IrisVerificationWorkflow] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.face_workflow = face_workflow or FaceVerificationWorkflow()
        self.iris_workflow = iris_workflow or IrisVerificationWorkflow()

        self.db = db_manager or DatabaseManager()
        self.db.initialize()

        # Score-level fusion parameters (config-driven, no longer hardcoded).
        fusion = get_config().fusion
        self.face_weight = fusion.face_weight
        self.iris_weight = fusion.iris_weight
        self.combined_threshold = fusion.combined_threshold

    @staticmethod
    def _normalize(confidence: float) -> float:
        """Map a 0-100 per-factor confidence onto a [0, 1] fusion score."""
        return max(0.0, min(1.0, confidence / 100.0))

    def verify(self, user_id: str) -> MultiModalVerificationResult:
        """Run face then iris verification and fuse the scores.

        The face step short-circuits the (more expensive) iris step on failure,
        since an AND gate can never recover from a failed factor. Real per-factor
        scores are always recorded.
        """

        logger.info("=" * 60)
        logger.info("STEP 1: FACE VERIFICATION")
        logger.info("=" * 60)

        face_result = self.face_workflow.verify(user_id)
        face_norm = self._normalize(face_result.confidence)

        if not face_result.success:
            self.db.log_verification(
                user_id=user_id,
                verification_type="combined",
                success=False,
                face_score=face_result.face_score,
                combined_score=self.face_weight * face_norm,
            )
            return MultiModalVerificationResult(
                success=False,
                user_id=user_id,
                face_success=False,
                iris_success=False,
                face_confidence=face_result.confidence,
                iris_confidence=0.0,
                combined_confidence=0.0,
                message="Face verification failed",
            )

        logger.info("=" * 60)
        logger.info("STEP 2: IRIS VERIFICATION")
        logger.info("=" * 60)

        iris_result = self.iris_workflow.verify(user_id)
        iris_norm = self._normalize(iris_result.confidence)

        # Weighted score-level fusion on the [0, 1] scale.
        combined_norm = self.face_weight * face_norm + self.iris_weight * iris_norm

        # AND gate: both factors must individually pass AND the fused score must
        # clear the combined threshold.
        success = (
            combined_norm >= self.combined_threshold
            and face_result.success
            and iris_result.success
        )

        if success:
            message = "Authentication Successful"
        elif not iris_result.success:
            message = "Iris verification failed"
        else:
            message = (
                f"Combined score {combined_norm:.2f} below "
                f"threshold {self.combined_threshold:.2f}"
            )

        self.db.log_verification(
            user_id=user_id,
            verification_type="combined",
            success=success,
            face_score=face_result.face_score,
            iris_score=iris_result.left_distance,
            combined_score=combined_norm,
        )

        return MultiModalVerificationResult(
            success=success,
            user_id=user_id,
            face_success=face_result.success,
            iris_success=iris_result.success,
            face_confidence=face_result.confidence,
            iris_confidence=iris_result.confidence,
            combined_confidence=combined_norm * 100.0,
            message=message,
        )
