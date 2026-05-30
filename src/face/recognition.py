"""
Face recognition module - detection, encoding, and matching.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import face_recognition

from src.utils.config import get_config


@dataclass
class FaceLocation:
    """Bounding box for a detected face."""
    top: int
    right: int
    bottom: int
    left: int
    
    @property
    def width(self) -> int:
        return self.right - self.left
    
    @property
    def height(self) -> int:
        return self.bottom - self.top
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (top, right, bottom, left) tuple."""
        return (self.top, self.right, self.bottom, self.left)
    
    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int]) -> "FaceLocation":
        """Create from (top, right, bottom, left) tuple."""
        return cls(top=t[0], right=t[1], bottom=t[2], left=t[3])


@dataclass
class DetectedFace:
    """A detected face with location and optional encoding."""
    location: FaceLocation
    encoding: Optional[np.ndarray] = None
    landmarks: Optional[dict] = None
    confidence: float = 1.0


@dataclass
class FaceTemplate:
    """A face template for enrollment/verification."""
    user_id: str
    encodings: List[np.ndarray] = field(default_factory=list)
    created_at: float = 0.0
    
    @property
    def num_samples(self) -> int:
        return len(self.encodings)
    
    def get_average_encoding(self) -> Optional[np.ndarray]:
        """Get the average of all encodings."""
        if not self.encodings:
            return None
        return np.mean(self.encodings, axis=0)


@dataclass
class MatchResult:
    """Result of a face matching operation."""
    is_match: bool
    user_id: Optional[str] = None
    similarity: float = 0.0
    
    @property
    def confidence(self) -> float:
        """Get confidence as percentage (0-100)."""
        return max(0.0, min(100.0, self.similarity * 100))


class FaceDetector:
    """
    Face detection using face_recognition library.
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize face detector.
        
        Args:
            model: Detection model - "cnn" (GPU, accurate) or "hog" (CPU, fast).
        """
        self.config = get_config()
        self.model = model or self.config.face_recognition.detection_model
        self.min_face_size = self.config.face_recognition.min_face_size
        
        print(f"FaceDetector initialized with model: {self.model}")
    
    def detect_faces(
        self,
        image: np.ndarray,
        num_upsample: int = 1
    ) -> List[FaceLocation]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image from OpenCV.
            num_upsample: Number of times to upsample image for smaller faces.
            
        Returns:
            List of FaceLocation objects.
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        cv2.imwrite("debug_frame.jpg", image)
        print("Saved debug frame")
        locations = face_recognition.face_locations(
            rgb_image,
            number_of_times_to_upsample=num_upsample,
            model=self.model
        )
        
        # Filter by minimum size
        face_locations = []
        for loc in locations:
            face_loc = FaceLocation.from_tuple(loc)
            if face_loc.width >= self.min_face_size and face_loc.height >= self.min_face_size:
                face_locations.append(face_loc)
        
        return face_locations
    
    def detect_largest_face(
        self,
        image: np.ndarray,
        num_upsample: int = 1
    ) -> Optional[FaceLocation]:
        """
        Detect and return only the largest face.
        
        Args:
            image: BGR image from OpenCV.
            num_upsample: Number of times to upsample.
            
        Returns:
            Largest FaceLocation or None if no faces found.
        """
        faces = self.detect_faces(image, num_upsample)
        
        if not faces:
            return None
        
        return max(faces, key=lambda f: f.area)


class FaceEncoder:
    """
    Face encoding using face_recognition library.
    """
    
    def __init__(self):
        """Initialize face encoder."""
        self.config = get_config()
        self.num_jitters = self.config.face_recognition.num_jitters
        self.model = self.config.face_recognition.encoding_model
        
        print(f"FaceEncoder initialized: jitters={self.num_jitters}, model={self.model}")
    
    def encode_face(
        self,
        image: np.ndarray,
        face_location: Optional[FaceLocation] = None
    ) -> Optional[np.ndarray]:
        """
        Generate 128-dimensional encoding for a face.
        
        Args:
            image: BGR image from OpenCV.
            face_location: Optional known face location. If None, will detect.
            
        Returns:
            128-D numpy array encoding, or None if no face found.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face locations
        if face_location is not None:
            known_locations = [face_location.to_tuple()]
        else:
            known_locations = None
        
        # Generate encodings
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=known_locations,
            num_jitters=self.num_jitters,
            model=self.model
        )
        
        if not encodings:
            return None
        
        return encodings[0]
    
    def encode_faces(
        self,
        image: np.ndarray,
        face_locations: Optional[List[FaceLocation]] = None
    ) -> List[np.ndarray]:
        """
        Generate encodings for all faces in an image.
        
        Args:
            image: BGR image from OpenCV.
            face_locations: Optional list of known face locations.
            
        Returns:
            List of 128-D encodings.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        known_locations = None
        if face_locations:
            known_locations = [loc.to_tuple() for loc in face_locations]
        
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=known_locations,
            num_jitters=self.num_jitters,
            model=self.model
        )
        
        return encodings


class FaceMatcher:
    """
    Face matching/comparison using face encodings.
    """
    
    def __init__(self, tolerance: Optional[float] = None):
        """
        Initialize face matcher.
        
        Args:
            tolerance: Match tolerance (0.0-1.0). Lower = stricter.
        """
        self.config = get_config()
        self.tolerance = tolerance or self.config.face_recognition.match_tolerance
        
        print(f"FaceMatcher initialized: tolerance={self.tolerance}")
    
    def compare(
        self,
        known_encoding: np.ndarray,
        unknown_encoding: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Compare two face encodings.
        
        Args:
            known_encoding: Reference encoding.
            unknown_encoding: Encoding to check.
            
        Returns:
            Tuple of (is_match, distance).
        """
        distance = float(face_recognition.face_distance([known_encoding], unknown_encoding)[0])
        is_match = distance <= self.tolerance
        
        return is_match, distance
    
    def compare_to_template(
        self,
        template: FaceTemplate,
        unknown_encoding: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Compare an encoding to a face template.
        
        Args:
            template: FaceTemplate with multiple encodings.
            unknown_encoding: Encoding to check.
            
        Returns:
            Tuple of (is_match, min_distance, avg_distance).
        """
        if not template.encodings:
            return False, 1.0, 1.0
        
        distances = face_recognition.face_distance(template.encodings, unknown_encoding)
        
        min_distance = float(np.min(distances))
        avg_distance = float(np.mean(distances))

        print("DEBUG MATCHER")
        print("min_distance =", min_distance)
        print("tolerance =", self.tolerance)

        is_match = min_distance <= self.tolerance

        print("is_match =", is_match)
        
        is_match = min_distance <= self.tolerance
        
        return is_match, min_distance, avg_distance
    
    def find_best_match(
        self,
        templates: List[FaceTemplate],
        unknown_encoding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching template from a list.
        
        Args:
            templates: List of FaceTemplates to search.
            unknown_encoding: Encoding to match.
            
        Returns:
            Tuple of (user_id or None, best_distance).
        """
        best_user_id = None
        best_distance = float('inf')
        
        for template in templates:
            is_match, min_dist, _ = self.compare_to_template(template, unknown_encoding)
            
            if min_dist < best_distance:
                best_distance = min_dist
                if is_match:
                    best_user_id = template.user_id
        
        return best_user_id, best_distance
    
    def get_similarity_score(self, distance: float) -> float:
        """
        Convert distance to a similarity score (0-100%).
        
        Args:
            distance: Euclidean distance (0.0 to ~1.0+).
            
        Returns:
            Similarity percentage (0-100).
        """
        # Distance of 0 = 100% similar, distance of tolerance = 50% similar
        similarity = max(0.0, (1.0 - distance / self.tolerance) * 50 + 50)
        return min(100.0, similarity)


class FaceRecognitionSystem:
    """
    High-level face recognition system combining detection, encoding, and matching.
    """
    
    def __init__(self):
        """Initialize the face recognition system."""
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.matcher = FaceMatcher()
        
        print("FaceRecognitionSystem initialized")
    
    def detect_and_encode(
        self,
        image: np.ndarray
    ) -> List[DetectedFace]:
        """
        Detect faces and generate encodings.
        
        Args:
            image: BGR image from OpenCV.
            
        Returns:
            List of DetectedFace objects with locations and encodings.
        """
        # Detect faces
        locations = self.detector.detect_faces(image)
        
        if not locations:
            return []
        
        # Encode faces
        encodings = self.encoder.encode_faces(image, locations)
        
        # Combine
        detected_faces = []
        for loc, enc in zip(locations, encodings):
            detected_faces.append(DetectedFace(
                location=loc,
                encoding=enc
            ))
        
        return detected_faces
    
    def get_largest_face(self, faces: List[DetectedFace]) -> Optional[DetectedFace]:
        """
        Get the largest face from a list of detected faces.
        
        Args:
            faces: List of detected faces.
            
        Returns:
            Largest DetectedFace or None if list is empty.
        """
        if not faces:
            return None
        
        return max(faces, key=lambda f: f.location.area)
    
    def verify(
        self,
        image: np.ndarray,
        template: FaceTemplate
    ) -> Tuple[bool, float, Optional[DetectedFace]]:
        """
        Verify a face against a template.
        
        Args:
            image: BGR image to verify.
            template: FaceTemplate to verify against.
            
        Returns:
            Tuple of (is_match, distance, detected_face).
        """
        # Detect and encode
        detected = self.detect_and_encode(image)
        
        if not detected:
            return False, 1.0, None
        
        # Use largest face
        face = self.get_largest_face(detected)
        
        # Compare
        is_match, min_dist, _ = self.matcher.compare_to_template(template, face.encoding)
        
        return is_match, min_dist, face
    
    def identify(
        self,
        image: np.ndarray,
        templates: List[FaceTemplate]
    ) -> Tuple[Optional[str], float, Optional[DetectedFace]]:
        """
        Identify a face from a list of templates.
        
        Args:
            image: BGR image to identify.
            templates: List of enrolled FaceTemplates.
            
        Returns:
            Tuple of (user_id or None, distance, detected_face).
        """
        # Detect and encode
        detected = self.detect_and_encode(image)
        
        if not detected:
            return None, 1.0, None
        
        # Use largest face
        face = self.get_largest_face(detected)
        
        # Find best match
        user_id, distance = self.matcher.find_best_match(templates, face.encoding)
        
        return user_id, distance, face
    
    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Draw face detections on an image.
        
        Args:
            image: BGR image to draw on.
            faces: List of detected faces.
            labels: Optional labels for each face.
            
        Returns:
            Image with drawn detections.
        """
        output = image.copy()
        
        for i, face in enumerate(faces):
            loc = face.location
            
            # Draw rectangle
            cv2.rectangle(
                output,
                (loc.left, loc.top),
                (loc.right, loc.bottom),
                (0, 255, 0),
                2
            )
            
            # Draw label
            label = labels[i] if labels and i < len(labels) else f"Face {i+1}"
            cv2.putText(
                output,
                label,
                (loc.left, loc.top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        return output


def test_face_recognition():
    """Test face recognition with webcam."""
    from src.capture.camera import Camera
    
    print("Testing face recognition...")
    system = FaceRecognitionSystem()
    
    with Camera() as cam:
        print("\nPress SPACE to capture and detect faces, Q to quit")
        
        while True:
            result = cam.read_frame()
            
            if not result.success:
                continue
            
            # Detect faces
            faces = system.detect_and_encode(result.frame)
            
            # Draw detections
            display = system.draw_detections(result.frame, faces)
            
            # Show face count
            cv2.putText(
                display,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Face Detection Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') and faces:
                print(f"\n✓ Detected {len(faces)} face(s)")
                for i, face in enumerate(faces):
                    print(f"  Face {i+1}: {face.location.width}x{face.location.height} at {face.location.center}")
                    if face.encoding is not None:
                        print(f"    Encoding: {face.encoding.shape} (first 5: {face.encoding[:5]})")
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_face_recognition()
