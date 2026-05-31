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
from typing import Optional

from ..capture.camera import Camera
from ..database.storage import DatabaseManager, EnrollmentResult, UserRecord
from ..face.recognition import FaceRecognitionSystem
from ..iris.recognition import IrisRecognitionSystem
from ..utils.config import get_config
from .enrollment import FaceEnrollmentWorkflow
from .verification import FaceVerificationWorkflow, VerificationResult
from .iris_verification import IrisVerificationWorkflow
from .multimodelauth import MultiModalVerificationWorkflow, MultiModalVerificationResult

logger = logging.getLogger(__name__)


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
