import cv2
import numpy as np
import face_recognition
from dataclasses import dataclass
from typing import Optional


@dataclass
class IrisTemplate:
    eye: str
    features: np.ndarray
    quality_score: float


class IrisRecognitionSystem:

    def extract_eye_regions(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks_list = face_recognition.face_landmarks(rgb)

        if not landmarks_list:
            return None

        landmarks = landmarks_list[0]

        left_eye = np.array(landmarks["left_eye"])
        right_eye = np.array(landmarks["right_eye"])

        return {
            "left_eye": left_eye,
            "right_eye": right_eye
        }

    def crop_eye(self, frame: np.ndarray, eye_points: np.ndarray):
        x, y, w, h = cv2.boundingRect(eye_points)

        padding = 8

        x = max(0, x - padding)
        y = max(0, y - padding)

        return frame[y:y+h+padding, x:x+w+padding]

    def generate_iris_template(self, eye_image):
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        gray = cv2.equalizeHist(gray)

        feature_vector = gray.flatten().astype("float32") / 255.0
        return feature_vector

    def compare_iris_templates(self, template1, template2):
        distance = ((template1 - template2) ** 2).mean() ** 0.5
        return distance








