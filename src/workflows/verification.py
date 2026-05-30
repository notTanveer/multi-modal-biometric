"""Face verification workflow module.

Provides functionality to verify users against enrolled biometric templates.
Supports both 1:1 verification (claim identity) and 1:N identification.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import cv2
import numpy as np

from ..capture.camera import Camera
from ..face.recognition import (
    FaceRecognitionSystem,
    FaceTemplate,
    MatchResult,
)
from ..database.storage import DatabaseManager, FaceTemplateRecord
from ..utils.config import get_config


class VerificationMode(Enum):
    """Verification mode options."""
    VERIFY = "verify"      # 1:1 verification against claimed identity
    IDENTIFY = "identify"  # 1:N identification against all enrolled users


@dataclass
class VerificationResult:
    """Result of a verification attempt."""
    
    success: bool
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    confidence: float = 0.0
    face_score: float = 0.0
    mode: VerificationMode = VerificationMode.VERIFY
    message: str = ""
    processing_time_ms: float = 0.0


class FaceVerificationWorkflow:
    """Handles face verification and identification.
    
    This class manages:
    1. 1:1 Verification - Verify a claimed identity
    2. 1:N Identification - Identify who a person is
    
    Usage:
        workflow = FaceVerificationWorkflow()
        
        # Verify a specific user
        result = workflow.verify("user123")
        
        # Identify who is in front of camera
        result = workflow.identify()
        
        # Continuous verification mode
        for result in workflow.continuous_verify("user123"):
            if result.success:
                print("Access granted!")
                break
    """
    
    def __init__(
        self,
        camera: Optional[Camera] = None,
        face_system: Optional[FaceRecognitionSystem] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        """Initialize verification workflow.
        
        Args:
            camera: Camera instance (creates new if None)
            face_system: Face recognition system (creates new if None)
            db_manager: Database manager (creates new if None)
        """
        self.config = get_config()
        
        self.camera = camera or Camera()
        self.face_system = face_system or FaceRecognitionSystem()
        self.db = db_manager or DatabaseManager()
        
        # Ensure database is initialized
        self.db.initialize()
        
        # Verification parameters
        self.face_threshold = self.config.face_recognition.match_tolerance
        self.max_attempts = self.config.verification.max_attempts
        self.timeout = self.config.verification.timeout_seconds
        
        # Cache for enrolled templates (optimization)
        self._template_cache: Optional[list[FaceTemplate]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 60.0  # Refresh cache every 60 seconds
    
    def _get_enrolled_templates(self, force_refresh: bool = False) -> list[FaceTemplate]:
        """Get all enrolled face templates (with caching).
        
        Args:
            force_refresh: Force cache refresh
            
        Returns:
            List of FaceTemplate objects
        """
        current_time = time.time()
        
        if (
            force_refresh or 
            self._template_cache is None or 
            (current_time - self._cache_time) > self._cache_ttl
        ):
            # Refresh cache
            records = self.db.get_all_face_templates()
            self._template_cache = [
                FaceTemplate(
                    user_id=record.user_id,
                    encodings=record.encodings
                )
                for record in records
            ]
            self._cache_time = current_time
        
        return self._template_cache
    
    def _get_user_template(self, user_id: str) -> Optional[FaceTemplate]:
        """Get face template for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            FaceTemplate if found, None otherwise
        """
        record = self.db.get_face_template(user_id)
        if record is None:
            return None
        
        return FaceTemplate(
            user_id=record.user_id,
            encodings=record.encodings
        )
    
    def verify_frame(
        self,
        frame: np.ndarray,
        user_id: str
    ) -> VerificationResult:
        """Verify a single frame against a specific user.
        
        Args:
            frame: BGR image frame
            user_id: User ID to verify against
            
        Returns:
            VerificationResult with match details
        """
        start_time = time.time()
        cv2.imwrite("debug_verify.jpg", frame)
        print("Frame shape:", frame.shape)
        
        # Get user's template
        template = self._get_user_template(user_id)
        if template is None:
            return VerificationResult(
                success=False,
                mode=VerificationMode.VERIFY,
                message=f"User '{user_id}' not enrolled"
            )
        
        # Get user info
        user = self.db.get_user(user_id)
        user_name = user.name if user else user_id
        
        # Detect and encode face
        cv2.imshow("Debug Frame", frame)
        cv2.waitKey(1)

        locations = self.face_system.detector.detect_faces(frame)
        print("LOCATIONS =", len(locations))
        detected_faces = self.face_system.detect_and_encode(frame)
        print("Faces found:", len(detected_faces))
        if not detected_faces:
            return VerificationResult(
                success=False,
                user_id=user_id,
                user_name=user_name,
                mode=VerificationMode.VERIFY,
                message="No face detected"



            )
        
        # Get largest face
        detected_face = self.face_system.get_largest_face(detected_faces)
        
        # Compare against template
        is_match, distance, _ = self.face_system.verify(frame, template)
        print("DEBUG")
        print("is_match =", is_match)
        print("distance =", distance)
        print("threshold =", self.face_threshold)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate confidence (inverse of distance)
        confidence = max(0.0, (1.0 - distance / self.face_threshold) * 100)
        
        # Log verification attempt
        self.db.log_verification(
            user_id=user_id,
            verification_type='face',
            success=is_match,
            face_score=distance
        )
        
        return VerificationResult(
            success=is_match,
            user_id=user_id,
            user_name=user_name,
            confidence=confidence,
            face_score=distance,
            mode=VerificationMode.VERIFY,
            message=f"Verified as {user_name}" if is_match else f"Not {user_name} (distance: {distance:.3f})",
            processing_time_ms=processing_time
        )
    
    def identify_frame(self, frame: np.ndarray) -> VerificationResult:
        """Identify who is in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            VerificationResult with identified user (if any)
        """
        start_time = time.time()
        
        # Get all enrolled templates
        templates = self._get_enrolled_templates()
        if not templates:
            return VerificationResult(
                success=False,
                mode=VerificationMode.IDENTIFY,
                message="No users enrolled"
            )
        
        # Detect and encode face
        detected_faces = self.face_system.detect_and_encode(frame)
        if not detected_faces:
            return VerificationResult(
                success=False,
                mode=VerificationMode.IDENTIFY,
                message="No face detected"
            )
        
        # Get largest face
        detected_face = self.face_system.get_largest_face(detected_faces)
        
        # Find best match
        user_id, distance, _ = self.face_system.identify(frame, templates)
        
        processing_time = (time.time() - start_time) * 1000
        
        if user_id is None or distance >   self.face_threshold:
            self.db.log_verification(
                user_id=None,
                verification_type='face',
                success=False,
                face_score=distance
            )
            
            return VerificationResult(
                success=False,
                mode=VerificationMode.IDENTIFY,
                message="No matching user found",
                face_score=distance,
                processing_time_ms=processing_time
            )
        
        # Get user info
        user = self.db.get_user(user_id)
        user_name = user.name if user else user_id
        
        # Calculate confidence
        confidence = max(0.0, (1.0 - distance / self.face_threshold) * 100)
        
        self.db.log_verification(
            user_id=user_id,
            verification_type='face',
            success=True,
            face_score=distance
        )
        
        return VerificationResult(
            success=True,
            user_id=user_id,
            user_name=user_name,
            confidence=confidence,
            face_score=distance,
            mode=VerificationMode.IDENTIFY,
            message=f"Identified as {user_name}",
            processing_time_ms=processing_time
        )
    
    def verify(
        self,
        user_id: str,
        show_preview: bool = True,
        timeout: Optional[float] = None
    ) -> VerificationResult:
        """Interactive verification against a specific user.
        
        Args:
            user_id: User ID to verify against
            show_preview: Whether to show camera preview
            timeout: Verification timeout in seconds
            
        Returns:
            VerificationResult with verification status
        """
        timeout = timeout or self.timeout
        attempts = 0
        start_time = time.time()
        
        # Check if user exists
        user = self.db.get_user(user_id)
        if not user:
            return VerificationResult(
                success=False,
                user_id=user_id,
                mode=VerificationMode.VERIFY,
                message=f"User '{user_id}' not enrolled"
            )
        
        try:
            if not self.camera.open():
                return VerificationResult(
                    success=False,
                    message="Failed to open camera"
                )
            
            print(f"\n{'='*50}")
            print(f"Face Verification for: {user.name} ({user_id})")
            print(f"{'='*50}")
            print("Please look at the camera. Press 'q' to cancel.")
            print(f"{'='*50}\n")
            
            best_result: Optional[VerificationResult] = None
            
            while attempts < self.max_attempts:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    return best_result or VerificationResult(
                        success=False,
                        user_id=user_id,
                        user_name=user.name,
                        mode=VerificationMode.VERIFY,
                        message=f"Verification timeout ({timeout}s)"
                    )
                
                # Read frame
                capture_result = self.camera.read_frame()
                if not capture_result.success or capture_result.frame is None:
                    continue
                
                frame = capture_result.frame
                
                # Verify frame

                result = self.verify_frame(frame, user_id)

                print("VERIFY_FRAME RESULT =", result)

                attempts += 1
                
                # Track best result
                if best_result is None or result.face_score > best_result.face_score:
                    best_result = result
                
                # Create preview
                preview_frame = frame.copy()
                
                # Draw status
                color = (0, 255, 0) if result.success else (0, 0, 255)
                status = "✓ VERIFIED" if result.success else "✗ NOT VERIFIED"
                
                cv2.putText(
                    preview_frame,
                    f"{status} ({result.face_score:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                
                cv2.putText(
                    preview_frame,
                    f"Attempt {attempts}/{self.max_attempts}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Draw face box if detected
                detect_faces = self.face_system.detect_and_encode(frame)

                if detect_faces:
                    detected_face = self.face_system.get_largest_face(detect_faces)
                    preview_frame = self.face_system.draw_detections(
                        preview_frame,
                        [detected_face]
                    )
                
                if show_preview:
                    cv2.imshow("Face Verification", preview_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        return VerificationResult(
                            success=False,
                            user_id=user_id,
                            user_name=user.name,
                            mode=VerificationMode.VERIFY,
                            message="Verification cancelled by user"
                        )
                
                # Success - return immediately
                if result.success:
                    print(f"\n✓ VERIFICATION PASSED")
                    print(f"  User: {result.user_name} ({result.user_id})")
                    print(f"  Match distance: {result.face_score:.3f}")
                    print(f"  Confidence: {result.confidence:.1f}%")
                    print(f"  Attempts: {attempts}/{self.max_attempts}")
                    time.sleep(0.5)  # Brief pause to show success
                    return result
                
                print(f"  Attempt {attempts}: distance={result.face_score:.3f}, not a match")
            
            # Max attempts reached
            print(f"\n✗ VERIFICATION FAILED")
            print(f"  User: {user.name} ({user_id})")
            print(f"  Failed after {attempts} attempts")
            if best_result and best_result.face_score > 0:
                print(f"  Best distance: {best_result.face_score:.3f} (threshold: {self.face_threshold})")
            return best_result or VerificationResult(
                success=False,
                user_id=user_id,
                user_name=user.name,
                mode=VerificationMode.VERIFY,
                message="Maximum attempts exceeded"
            )
            
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
    
    def identify(
        self,
        show_preview: bool = True,
        timeout: Optional[float] = None
    ) -> VerificationResult:
        """Interactive identification (1:N).
        
        Args:
            show_preview: Whether to show camera preview
            timeout: Identification timeout in seconds
            
        Returns:
            VerificationResult with identified user (if any)
        """
        timeout = timeout or self.timeout
        attempts = 0
        start_time = time.time()
        
        try:
            if not self.camera.open():
                return VerificationResult(
                    success=False,
                    mode=VerificationMode.IDENTIFY,
                    message="Failed to open camera"
                )
            
            print(f"\n{'='*50}")
            print("Face Identification")
            print(f"{'='*50}")
            print("Please look at the camera. Press 'q' to cancel.")
            print(f"{'='*50}\n")
            
            best_result: Optional[VerificationResult] = None
            
            while attempts < self.max_attempts:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    return best_result or VerificationResult(
                        success=False,
                        mode=VerificationMode.IDENTIFY,
                        message=f"Identification timeout ({timeout}s)"
                    )
                
                # Read frame
                capture_result = self.camera.read_frame()
                if not capture_result.success or capture_result.frame is None:
                    continue
                
                frame = capture_result.frame
                
                # Identify frame
                result = self.identify_frame(frame)
                attempts += 1
                
                # Track best result
                if best_result is None or result.face_score < best_result.face_score:
                    best_result = result
                
                # Create preview
                preview_frame = frame.copy()
                
                # Draw status
                if result.success:
                    color = (0, 255, 0)
                    status = f"✓ {result.user_name}"
                else:
                    color = (0, 165, 255)
                    status = "Searching..."
                
                cv2.putText(
                    preview_frame,
                    f"{status} ({result.face_score:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                
                # Draw face box if detected
                detect_faces = self.face_system.detect_and_encode(frame)
                if detect_faces:
                    preview_frame = self.face_system.draw_detections(
                        preview_frame,
                        detect_faces
                    )
                
                if show_preview:
                    cv2.imshow("Face Identification", preview_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        return VerificationResult(
                            success=False,
                            mode=VerificationMode.IDENTIFY,
                            message="Identification cancelled by user"
                        )
                
                # Success - return immediately
                if result.success:
                    print(f"\n✓ MATCH FOUND")
                    print(f"  User: {result.user_name} ({result.user_id})")
                    print(f"  Match distance: {result.face_score:.3f}")
                    print(f"  Confidence: {result.confidence:.1f}%")
                    time.sleep(0.5)
                    return result
                
                print(f"  Attempt {attempts}: {result.message}")
            
            # Max attempts reached
            print(f"✗ Identification failed after {attempts} attempts")
            return best_result or VerificationResult(
                success=False,
                mode=VerificationMode.IDENTIFY,
                message="Could not identify user"
            )
            
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
    
    def continuous_verify(
        self,
        user_id: str,
        show_preview: bool = True
    ):
        """Generator for continuous verification.
        
        Yields verification results continuously until stopped.
        Useful for real-time access control.
        
        Args:
            user_id: User ID to verify against
            show_preview: Whether to show preview
            
        Yields:
            VerificationResult for each frame
        """
        try:
            if not self.camera.open():
                yield VerificationResult(
                    success=False,
                    message="Failed to open camera"
                )
                return
            
            while True:
                capture_result = self.camera.read_frame()
                if not capture_result.success or capture_result.frame is None:
                    continue
                
                frame = capture_result.frame
                result = self.verify_frame(frame, user_id)
                
                if show_preview:
                    preview_frame = frame.copy()
                    color = (0, 255, 0) if result.success else (0, 0, 255)
                    
                    cv2.putText(
                        preview_frame,
                        f"Score: {result.face_score:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    
                    cv2.imshow("Continuous Verification", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                yield result
                
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
    
    def continuous_identify(self, show_preview: bool = True):
        """Generator for continuous identification.
        
        Yields identification results continuously.
        
        Args:
            show_preview: Whether to show preview
            
        Yields:
            VerificationResult for each frame
        """
        try:
            if not self.camera.open():
                yield VerificationResult(
                    success=False,
                    mode=VerificationMode.IDENTIFY,
                    message="Failed to open camera"
                )
                return
            
            while True:
                capture_result = self.camera.read_frame()
                if not capture_result.success or capture_result.frame is None:
                    continue
                
                frame = capture_result.frame
                result = self.identify_frame(frame)
                
                if show_preview:
                    preview_frame = frame.copy()
                    
                    if result.success:
                        color = (0, 255, 0)
                        text = f"{result.user_name} ({result.face_score:.2f})"
                    else:
                        color = (0, 165, 255)
                        text = "Unknown"
                    
                    cv2.putText(
                        preview_frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    
                    cv2.imshow("Continuous Identification", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                yield result
                
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
