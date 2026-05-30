"""
Camera capture module for face and iris image acquisition.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
from dataclasses import dataclass
import time

from src.utils.config import get_config


@dataclass
class ImageQuality:
    """Image quality metrics."""
    brightness: float
    sharpness: float
    is_acceptable: bool
    issues: list


@dataclass  
class CaptureResult:
    """Result of a camera capture operation."""
    success: bool
    frame: Optional[np.ndarray]
    quality: Optional[ImageQuality]
    timestamp: float
    error: Optional[str] = None


class Camera:
    """
    Webcam interface for capturing images.
    
    Handles camera initialization, frame capture, and basic quality assessment.
    """
    
    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize camera.
        
        Args:
            device_id: Camera device ID. If None, uses config value.
        """
        self.config = get_config()
        self.device_id = device_id if device_id is not None else self.config.camera.device_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False

    def open(self) -> bool:
            """
            Open the camera device.
            """

            if self.is_open:
                return True

            # MOBILE CAMERA STREAM URL
            url = "http://172.30.168.162:8080/video"

            print(f"Connecting to mobile camera: {url}")

            self.cap = cv2.VideoCapture(url)

            if not self.cap.isOpened():
                print("Error: Could not open mobile camera")
                return False

            # Test frame
            ret, frame = self.cap.read()

            if not ret or frame is None:
                print("Error: Could not read frame")
                self.cap.release()
                return False

            self.is_open = True

            print("Mobile camera connected successfully")

            return True
    
    def close(self) -> None:
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            print("Camera closed")
    
    def read_frame(self) -> CaptureResult:
        """
        Capture a single frame.
        
        Returns:
            CaptureResult with frame and quality info.
        """
        if not self.is_open:
            if not self.open():
                return CaptureResult(
                    success=False,
                    frame=None,
                    quality=None,
                    timestamp=time.time(),
                    error="Camera not open"
                )
        
        ret, frame = self.cap.read()
        timestamp = time.time()
        
        if not ret or frame is None:
            return CaptureResult(
                success=False,
                frame=None,
                quality=None,
                timestamp=timestamp,
                error="Failed to capture frame"
            )
        
        # Assess quality
        quality = self.assess_quality(frame)
        
        return CaptureResult(
            success=True,
            frame=frame,
            quality=quality,
            timestamp=timestamp
        )
    
    def assess_quality(self, frame: np.ndarray) -> ImageQuality:
        """
        Assess the quality of a captured frame.
        
        Args:
            frame: BGR image from camera.
            
        Returns:
            ImageQuality with metrics and acceptability.
        """
        issues = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness (mean pixel value)
        brightness = float(np.mean(gray))
        
        if brightness < self.config.quality.min_brightness:
            issues.append(f"Too dark (brightness: {brightness:.1f})")
        elif brightness > self.config.quality.max_brightness:
            issues.append(f"Too bright (brightness: {brightness:.1f})")
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        
        if sharpness < self.config.quality.min_sharpness:
            issues.append(f"Too blurry (sharpness: {sharpness:.1f})")
        
        is_acceptable = len(issues) == 0
        
        return ImageQuality(
            brightness=brightness,
            sharpness=sharpness,
            is_acceptable=is_acceptable,
            issues=issues
        )
    
    def capture_with_preview(
        self,
        window_name: str = "Camera Preview",
        instruction: str = "Press SPACE to capture, Q to quit"
    ) -> CaptureResult:
        """
        Show live preview and capture on keypress.
        
        Args:
            window_name: OpenCV window title.
            instruction: Text to show on screen.
            
        Returns:
            CaptureResult when user presses SPACE, or failure on Q.
        """
        if not self.is_open:
            if not self.open():
                return CaptureResult(
                    success=False,
                    frame=None,
                    quality=None,
                    timestamp=time.time(),
                    error="Camera not open"
                )
        
        print(f"\n{instruction}")
        
        while True:
            result = self.read_frame()
            
            if not result.success:
                continue
            
            # Draw overlay
            display_frame = result.frame.copy()
            self._draw_overlay(display_frame, result.quality, instruction)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - capture
                cv2.destroyWindow(window_name)
                return result
            elif key == ord('q') or key == 27:  # Q or Escape - quit
                cv2.destroyWindow(window_name)
                return CaptureResult(
                    success=False,
                    frame=None,
                    quality=None,
                    timestamp=time.time(),
                    error="Cancelled by user"
                )
    
    def stream_frames(self) -> Generator[CaptureResult, None, None]:
        """
        Generator that yields frames continuously.
        
        Yields:
            CaptureResult for each frame.
        """
        if not self.is_open:
            if not self.open():
                return
        
        while self.is_open:
            yield self.read_frame()
    
    def _draw_overlay(
        self,
        frame: np.ndarray,
        quality: ImageQuality,
        instruction: str
    ) -> None:
        """Draw quality info and instructions on frame."""
        h, w = frame.shape[:2]
        
        # Quality indicator
        color = (0, 255, 0) if quality.is_acceptable else (0, 0, 255)
        status = "GOOD" if quality.is_acceptable else "POOR"
        
        cv2.putText(
            frame, f"Quality: {status}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )
        
        # Quality metrics
        cv2.putText(
            frame, f"Brightness: {quality.brightness:.0f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )
        cv2.putText(
            frame, f"Sharpness: {quality.sharpness:.0f}",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )
        
        # Issues
        y_offset = 110
        for issue in quality.issues:
            cv2.putText(
                frame, f"! {issue}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1
            )
            y_offset += 25
        
        # Instructions at bottom
        cv2.putText(
            frame, instruction,
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def test_camera():
    """Quick camera test."""
    print("Testing camera...")
    
    with Camera() as cam:
        result = cam.capture_with_preview(
            instruction="Press SPACE to test capture, Q to quit"
        )
        
        if result.success:
            print(f"✓ Captured frame: {result.frame.shape}")
            print(f"  Brightness: {result.quality.brightness:.1f}")
            print(f"  Sharpness: {result.quality.sharpness:.1f}")
            print(f"  Quality OK: {result.quality.is_acceptable}")
        else:
            print(f"✗ Capture failed: {result.error}")


if __name__ == "__main__":
    test_camera()
