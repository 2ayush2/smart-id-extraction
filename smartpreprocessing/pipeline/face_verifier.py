import numpy as np
from insightface.app import FaceAnalysis
import cv2
import os
import logging
from typing import Tuple, Optional
from utils.preprocess import enhance_image

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_id_extraction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FaceVerifier:
    def __init__(self):
        try:
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            logger.debug("InsightFace initialized")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.face_cascade.empty():
                logger.error("Failed to load Haar Cascade classifier")
                raise RuntimeError("Failed to load Haar Cascade classifier")
            logger.debug("Haar Cascade loaded")
        except Exception as e:
            logger.error(f"FaceVerifier initialization failed: {str(e)}")
            raise

    def preprocess_for_detection(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for face detection."""
        try:
            max_size = 1280
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return img

    def detect_and_embed(self, image_path: str, output_dir: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Detect face and generate embedding."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None, None

            processed_img = self.preprocess_for_detection(img)
            faces = self.app.get(processed_img)
            if faces:
                face = faces[0]
                embedding = face.embedding
                bbox = face.bbox.astype(int)
                logger.debug(f"InsightFace detected face in {image_path}")
            else:
                logger.debug(f"InsightFace failed, trying Haar Cascade")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))
                if len(faces) == 0:
                    enhanced_img = enhance_image(img)
                    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))
                    if len(faces) == 0:
                        logger.warning(f"No face detected in {image_path}")
                        return None, None
                    img = enhanced_img
                (x, y, w, h) = faces[0]
                logger.debug(f"Haar Cascade detected face at bbox: x={x}, y={y}, w={w}, h={h}")
                bbox = [x, y, x + w, y + h]
                cropped_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                faces = self.app.get(cropped_face_rgb)
                if not faces:
                    logger.warning(f"Embedding failed for cropped face in {image_path}")
                    return None, None
                embedding = faces[0].embedding
                logger.debug(f"Haar Cascade detected face, embedding generated")

            face_path = os.path.join(output_dir, f"cropped_face_{os.path.basename(image_path)}")
            cv2.imwrite(face_path, img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            logger.debug(f"Cropped face saved to {face_path}")
            return embedding, face_path
        except Exception as e:
            logger.error(f"Face detection error in {image_path}: {str(e)}")
            return None, None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            similarity = float(np.clip(np.dot(embedding1, embedding2), 0.0, 1.0))
            logger.debug(f"Similarity score: {similarity}")
            return similarity
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            return 0.0

    def verify_faces(self, id_image_path: str, selfie_image_path: str, output_dir: str = None) -> float:
        """Verify if faces in two images match."""
        try:
            id_embedding, id_face_path = self.detect_and_embed(id_image_path, output_dir or os.path.dirname(id_image_path))
            if id_embedding is None:
                logger.warning("No face detected in ID image")
                return 0.0

            selfie_embedding, selfie_face_path = self.detect_and_embed(selfie_image_path, output_dir or os.path.dirname(selfie_image_path))
            if selfie_embedding is None:
                logger.warning("No face detected in selfie image")
                return 0.0

            match_score = self.compute_similarity(id_embedding, selfie_embedding)
            logger.debug(f"Verification score: {match_score}")
            return match_score
        except Exception as e:
            logger.error(f"Face verification failed: {str(e)}")
            return 0.0