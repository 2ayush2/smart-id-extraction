import cv2
import dlib
import numpy as np
import logging
import os
from typing import List, Tuple

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_id_extraction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LivenessChecker:
    def __init__(self, predictor_path: str = "shape_predictor_68_face_landmarks.dat"):
        try:
            self.detector = dlib.get_frontal_face_detector()
            if not os.path.exists(predictor_path):
                logger.error(f"Shape predictor file not found: {predictor_path}")
                raise FileNotFoundError(f"Shape predictor file not found: {predictor_path}")
            self.predictor = dlib.shape_predictor(predictor_path)
            self.haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.haar_cascade.empty():
                logger.error("Failed to load Haar Cascade classifier")
                raise FileNotFoundError("Haar Cascade classifier not found")
            logger.debug("LivenessChecker initialized")
        except Exception as e:
            logger.error(f"LivenessChecker initialization failed: {str(e)}")
            raise

    def _eye_aspect_ratio(self, eye: List[Tuple[int, int]]) -> float:
        try:
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            if C == 0:
                logger.warning("Eye width is zero, returning 0.0")
                return 0.0
            ear = (A + B) / (2.0 * C)
            return ear if not np.isnan(ear) else 0.0
        except Exception as e:
            logger.warning(f"EAR computation failed: {str(e)}")
            return 0.0

    def check_liveness(self, image_path: str = None, use_video: bool = False, ear_threshold: float = 0.20, frames_to_check: int = 30, fps: int = 10) -> bool:
        if use_video:
            return self._check_liveness_video(ear_threshold, frames_to_check, fps)
        else:
            if not image_path or not os.path.exists(image_path):
                logger.error(f"Invalid or missing image path: {image_path}")
                return False
            return self._check_liveness_static(image_path, ear_threshold)

    def _check_liveness_static(self, image_path: str, ear_threshold: float) -> bool:
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False

            max_size = 640
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) == 0:
                logger.debug("dlib face detection failed, trying Haar Cascade")
                faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3, minSize=(30, 30))
                if len(faces) == 0:
                    logger.warning(f"No face detected in {image_path}")
                    return False
                (x, y, w, h) = faces[0]
                faces = [dlib.rectangle(x, y, x + w, y + h)]

            face = faces[0]
            landmarks = self.predictor(gray, face)

            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            is_alive = avg_ear > ear_threshold
            logger.debug(f"Static liveness check: is_alive={is_alive}, avg_ear={avg_ear}")
            return is_alive

        except Exception as e:
            logger.error(f"Static liveness check failed: {str(e)}")
            return False

    def _check_liveness_video(self, ear_threshold: float, frames_to_check: int, fps: int) -> bool:
        try:
            cap = cv2.VideoCapture(0)  # Use webcam
            if not cap.isOpened():
                logger.error("Failed to open webcam")
                return False

            blink_count = 0
            ear_list = []
            frame_count = 0

            while frame_count < frames_to_check:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                if len(faces) > 0:
                    face = faces[0]
                    landmarks = self.predictor(gray, face)
                    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                    avg_ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0
                    ear_list.append(avg_ear)

                    if len(ear_list) > 2 and ear_list[-2] < ear_threshold and avg_ear > ear_threshold:
                        blink_count += 1

                frame_count += 1
                cv2.imshow("Liveness Check - Blink to Verify", frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            is_alive = blink_count >= 1  # At least one blink detected
            logger.debug(f"Video liveness check: is_alive={is_alive}, blink_count={blink_count}")
            return is_alive

        except Exception as e:
            logger.error(f"Video liveness check failed: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            return False