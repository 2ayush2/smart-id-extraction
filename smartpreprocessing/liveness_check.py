import cv2
import dlib
import numpy as np
import logging
from typing import List, Tuple, Optional
import threading
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_id_extraction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Thread-local storage for dlib components
thread_local = threading.local()

class LivenessChecker:
    def __init__(self):
        """Initialize LivenessChecker with thread-safe components."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            logger.error("Failed to load Haar Cascade classifier")
            raise RuntimeError("Failed to load Haar Cascade classifier")
        logger.debug("Haar Cascade loaded successfully")
        self.ear_threshold = 0.2  # Eye aspect ratio threshold for blink detection
        self.consecutive_frames = 3  # Number of frames to confirm a blink

    def get_dlib_components(self):
        """Get or create dlib components for the current thread."""
        if not hasattr(thread_local, "detector"):
            try:
                thread_local.detector = dlib.get_frontal_face_detector()
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                if not os.path.exists(predictor_path):
                    logger.error(f"Shape predictor file not found: {predictor_path}")
                    raise FileNotFoundError(f"Shape predictor file not found: {predictor_path}")
                thread_local.predictor = dlib.shape_predictor(predictor_path)
                logger.debug("Initialized dlib components for thread")
            except Exception as e:
                logger.error(f"dlib initialization failed: {str(e)}")
                thread_local.detector = None
                thread_local.predictor = None
        return thread_local.detector, thread_local.predictor

    def _eye_aspect_ratio(self, eye: List[Tuple[int, int]]) -> float:
        """Compute eye aspect ratio (EAR) for blink detection."""
        try:
            # Calculate Euclidean distances between vertical eye landmarks
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            # Calculate Euclidean distance between horizontal eye landmarks
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            # Compute EAR
            ear = (A + B) / (2.0 * C)
            return ear if not np.isnan(ear) else 0.0
        except Exception as e:
            logger.warning(f"EAR calculation failed: {str(e)}")
            return 0.0

    def _get_eye_landmarks(self, shape: dlib.full_object_detection) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Extract left and right eye landmarks from dlib shape."""
        try:
            left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            return left_eye, right_eye
        except Exception as e:
            logger.warning(f"Failed to extract eye landmarks: {str(e)}")
            return [], []

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for faster detection."""
        try:
            max_size = 480  # Reduced for speed
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return image

    def check_liveness(self, image_path: str) -> bool:
        """Check if the image contains a live face by detecting blinks."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False

            gray = self.preprocess_image(image)
            detector, predictor = self.get_dlib_components()
            if detector is None or predictor is None:
                logger.error("dlib components not initialized")
                return False

            # Detect faces
            faces = detector(gray, 0)
            if not faces:
                logger.warning(f"No faces detected in {image_path}")
                return False

            # Process the first detected face
            face = faces[0]
            shape = predictor(gray, face)
            left_eye, right_eye = self._get_eye_landmarks(shape)

            if not left_eye or not right_eye:
                logger.warning("Failed to detect eye landmarks")
                return False

            # Calculate EAR for both eyes
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            logger.debug(f"Average EAR: {avg_ear:.3f}")

            # For a static image, we assume liveness if EAR is above threshold
            # (indicating open eyes, as we can't detect blinks in a single frame)
            is_live = avg_ear > self.ear_threshold
            logger.debug(f"Liveness check result: {'Live' if is_live else 'Not Live'} (EAR: {avg_ear:.3f})")
            return is_live

        except Exception as e:
            logger.error(f"Liveness check failed for {image_path}: {str(e)}")
            return False