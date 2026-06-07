"""In-process facade over the biometric workflows.

The GUI used to shell out to ``demo/face_demo.py`` / ``test_fusionauth.py`` and
infer success by string-matching stdout. That was fragile and, worse, insecure:
the substring ``face_success=True`` appears in a *denied* multi-modal result, so
a face-pass / iris-fail was shown as authenticated.

``BiometricService`` exposes the workflows directly and returns the existing
structured result dataclasses, so callers decide on ``result.success`` rather
than scraped text. A single ``Camera``, ``DatabaseManager``, and recognition
systems are shared across operations.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..capture.camera import Camera
from ..database.storage import DatabaseManager, EnrollmentResult, UserRecord
from ..face.recognition import FaceRecognitionSystem, FaceTemplate
from ..iris.recognition import IrisRecognitionSystem
from ..utils.config import get_config
from .enrollment import FaceEnrollmentWorkflow
from .verification import FaceVerificationWorkflow, VerificationResult
from .iris_verification import IrisVerificationWorkflow
from .multimodelauth import MultiModalVerificationWorkflow, MultiModalVerificationResult
from .fusion import combine_scores, normalize

logger = logging.getLogger(__name__)


@dataclass
class FaceMatch:
    """Per-frame face-matching outcome (no camera, no GUI)."""
    found: bool = False
    is_match: bool = False
    distance: float = 1.0
    confidence: float = 0.0
    box: Optional[tuple] = None  # (left, top, right, bottom)


@dataclass
class IrisMatch:
    """Per-frame iris-matching outcome plus the eye-aspect-ratio for liveness."""
    found: bool = False
    is_match: bool = False
    distance: float = 999.0
    confidence: float = 0.0
    ear: float = 1.0
    quality_ok: bool = False


class BiometricService:
    """Single entry point the UI (or any caller) uses for all operations."""

    def __init__(self):
        self.config = get_config()
        self.db = DatabaseManager()
        self.db.initialize()

        # Shared so models load once and only one camera device is touched.
        self.camera = Camera()
        self.face_system = FaceRecognitionSystem()
        self.iris_system = IrisRecognitionSystem()

    # ------------------------------------------------------------------
    # Enrollment / verification
    # ------------------------------------------------------------------
    def enroll(
        self,
        user_id: str,
        name: str,
        num_samples: Optional[int] = None,
    ) -> EnrollmentResult:
        workflow = FaceEnrollmentWorkflow(
            camera=self.camera,
            face_system=self.face_system,
            db_manager=self.db,
        )
        workflow.iris_system = self.iris_system
        return workflow.enroll_interactive(user_id, name, num_samples=num_samples)

    def verify_face(self, user_id: str) -> VerificationResult:
        workflow = FaceVerificationWorkflow(
            camera=self.camera,
            face_system=self.face_system,
            db_manager=self.db,
        )
        return workflow.verify(user_id)

    def authenticate(self, user_id: str) -> MultiModalVerificationResult:
        """Full multi-modal (face + iris) authentication."""
        face_workflow = FaceVerificationWorkflow(
            camera=self.camera,
            face_system=self.face_system,
            db_manager=self.db,
        )
        iris_workflow = IrisVerificationWorkflow(
            camera=self.camera,
            iris_system=self.iris_system,
            db_manager=self.db,
        )
        workflow = MultiModalVerificationWorkflow(
            face_workflow=face_workflow,
            iris_workflow=iris_workflow,
            db_manager=self.db,
        )
        return workflow.verify(user_id)

    # ------------------------------------------------------------------
    # Frame-driven API (used by the GUI; no camera/windows owned here)
    #
    # These are stateless, single-frame computations. The caller owns the
    # camera/preview loop (on its main thread) and feeds frames in, so there
    # are no OpenCV windows and no cross-thread GUI calls.
    # ------------------------------------------------------------------
    def load_face_template(self, user_id: str) -> Optional[FaceTemplate]:
        record = self.db.get_face_template(user_id)
        if record is None:
            return None
        return FaceTemplate(user_id=record.user_id, encodings=record.encodings)

    def match_face(self, frame, template: FaceTemplate) -> FaceMatch:
        detected = self.face_system.detect_and_encode(frame)
        if not detected:
            return FaceMatch(found=False)

        face = self.face_system.get_largest_face(detected)
        is_match, distance, _ = self.face_system.matcher.compare_to_template(
            template, face.encoding
        )
        tol = max(self.face_system.matcher.tolerance, 1e-6)
        confidence = max(0.0, (1.0 - distance / tol) * 100)
        loc = face.location
        return FaceMatch(
            found=True,
            is_match=is_match,
            distance=distance,
            confidence=confidence,
            box=(loc.left, loc.top, loc.right, loc.bottom),
        )

    def load_iris_templates(self, user_id: str):
        left = self.db.get_iris_template(user_id, "left")
        right = self.db.get_iris_template(user_id, "right")
        if left is None or right is None:
            return None
        return left, right

    def match_iris(self, frame, left_template, right_template) -> IrisMatch:
        eyes = self.iris_system.extract_eye_regions(frame)
        if eyes is None:
            return IrisMatch(found=False)

        ear = self.iris_system.average_ear(eyes)
        left_crop = self.iris_system.crop_eye(frame, eyes["left_eye"])
        right_crop = self.iris_system.crop_eye(frame, eyes["right_eye"])
        quality_ok = (
            self.iris_system.is_acceptable(left_crop)
            and self.iris_system.is_acceptable(right_crop)
        )

        # Only template/compare on good crops. On poor frames (mid-blink, eye at
        # the edge) still report the eyes + EAR so the caller's blink/liveness
        # state machine keeps advancing — just don't claim a match.
        if not quality_ok:
            return IrisMatch(found=True, ear=ear, quality_ok=False)

        live_left = self.iris_system.generate_iris_template(left_crop)
        live_right = self.iris_system.generate_iris_template(right_crop)
        left_d = self.iris_system.compare_iris_templates(live_left, left_template)
        right_d = self.iris_system.compare_iris_templates(live_right, right_template)
        avg = (left_d + right_d) / 2

        thr = max(self.iris_system.threshold, 1e-6)
        confidence = max(0.0, (1 - avg / thr) * 100)
        return IrisMatch(
            found=True,
            is_match=avg < self.iris_system.threshold,
            distance=avg,
            confidence=confidence,
            ear=ear,
            quality_ok=quality_ok,
        )

    def fuse(self, face_confidence: float, iris_confidence: float):
        """Weighted score-level fusion. Returns ``(combined_norm, combined_pct)``."""
        f = self.config.fusion
        combined_norm = combine_scores(
            normalize(face_confidence),
            normalize(iris_confidence),
            f.face_weight,
            f.iris_weight,
        )
        return combined_norm, combined_norm * 100.0

    def collect_iris_sample(self, frame, left_bucket: list, right_bucket: list, num_samples: int) -> bool:
        """Append one good-quality template per eye to the buckets if possible.

        Returns True when both buckets have reached ``num_samples``.
        """
        eyes = self.iris_system.extract_eye_regions(frame)
        if eyes is not None:
            for name, bucket in (("left_eye", left_bucket), ("right_eye", right_bucket)):
                if len(bucket) >= num_samples:
                    continue
                crop = self.iris_system.crop_eye(frame, eyes[name])
                if self.iris_system.is_acceptable(crop):
                    bucket.append(self.iris_system.generate_iris_template(crop))
        return len(left_bucket) >= num_samples and len(right_bucket) >= num_samples

    def make_enrollment_workflow(self) -> FaceEnrollmentWorkflow:
        """An enrollment workflow wired to the shared systems (no camera use)."""
        workflow = FaceEnrollmentWorkflow(
            camera=self.camera,
            face_system=self.face_system,
            db_manager=self.db,
        )
        workflow.iris_system = self.iris_system
        return workflow

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------
    def list_users(self) -> list[UserRecord]:
        return self.db.list_users(active_only=False)

    def delete_user(self, user_id: str) -> tuple[bool, str]:
        """Delete a user and all associated templates.

        Returns ``(ok, message)``. Replaces the old bare ``except: pass`` flow
        with specific, logged handling.
        """
        user = self.db.get_user(user_id)
        if user is None:
            return False, f"User '{user_id}' not found"

        try:
            self.db.delete_face_template(user_id)
            self.db.delete_iris_template(user_id)
            self.db.delete_user(user_id)
        except Exception as exc:  # storage-layer failure
            logger.exception("Failed to delete user %s", user_id)
            return False, f"Error deleting '{user_id}': {exc}"

        return True, f"Deleted user: {user_id}"
