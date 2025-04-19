import os
import cv2
import json
import numpy as np
import logging
from typing import Dict, Optional, Any, List

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("selfie_ocr.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to align with app.py
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, bool):
            return str(obj).lower()
        return super().default(obj)

def enhance_image(image: np.ndarray) -> np.ndarray:
    try:
        if image.size == 0:
            logger.error("Empty image provided to enhance_image")
            return image
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        logger.debug("Image enhanced")
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image

def detect_faces(image: np.ndarray) -> List[Dict[str, Any]]:
    try:
        from utils.preprocess import FACE_APP
        if FACE_APP is None:
            logger.error("InsightFace not initialized")
            return []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = FACE_APP.get(rgb_image)
        logger.debug(f"Detected {len(faces)} faces")
        return [
            {
                "x1": int(face.bbox[0]),
                "y1": int(face.bbox[1]),
                "x2": int(face.bbox[2]),
                "y2": int(face.bbox[3]),
                "size": (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
                "det_score": face.det_score
            } for face in faces
        ]
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []

def save_results(result: Dict[str, Any], annotated_image: np.ndarray, output_dir: str, source_image: str) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"result_{os.path.splitext(os.path.basename(source_image))[0]}.json").replace("\\", "/")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False, cls=CustomJSONEncoder)
        logger.debug(f"Saved results to {json_path}")
        annotated_path = os.path.join(output_dir, f"annotated_{os.path.basename(source_image)}").replace("\\", "/")
        cv2.imwrite(annotated_path, annotated_image)
        logger.debug(f"Saved annotated image to {annotated_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def save_face_crop(image: np.ndarray, face: Dict[str, Any], output_dir: str, source_image: str) -> str:
    try:
        crops_dir = os.path.join(output_dir, "crops").replace("\\", "/")
        os.makedirs(crops_dir, exist_ok=True)
        x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("Empty face crop")
            return ""
        crop_path = os.path.join(crops_dir, f"selfie_face_{os.path.basename(source_image)}").replace("\\", "/")
        cv2.imwrite(crop_path, crop)
        logger.debug(f"Saved face crop to {crop_path}")
        return crop_path
    except Exception as e:
        logger.error(f"Error saving face crop: {str(e)}")
        return ""

def process_image(img_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
    logger.debug(f"Processing Selfie image: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        logger.error(f"Failed to load image {img_path}")
        return {"fields": {"error": {"ocr_text": "Invalid image", "confidence": 0.0}}, "confidences": {"error": 0.0}}
    if image.shape[0] < 100 or image.shape[1] < 100:
        logger.warning(f"Image too small: {img_path}")
        return {"fields": {"error": {"ocr_text": "Image too small", "confidence": 0.0}}, "confidences": {"error": 0.0}}
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    file_name = os.path.basename(img_path)
    annotated = image.copy()
    fields = {}
    confidences = {}

    try:
        # Enhance image
        enhanced = enhance_image(image)
        # Detect faces
        faces = detect_faces(enhanced)
        logger.debug(f"Faces detected: {len(faces)}, details: {faces}")
        crop_path = ""
        confidence = 0.0

        # Select highest-confidence face
        if faces:
            face = max(faces, key=lambda f: f["det_score"])
            confidence = min(float(face["det_score"]), 0.95)
            cv2.rectangle(annotated, (face["x1"], face["y1"]), (face["x2"], face["y2"]), (0, 255, 0), 2)
            cv2.putText(annotated, f"Face ({confidence:.2f})", (face["x1"], face["y1"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            crop_path = save_face_crop(image, face, output_dir, file_name)

        # Prepare result
        fields["Selfie"] = {
            "ocr_text": "Face detected" if faces else "No face detected",
            "confidence": confidence,
            "crop_image_path": crop_path,
            "source_image": file_name
        }
        confidences["Selfie"] = confidence

        if not faces:
            fields["error"] = {
                "ocr_text": "No face detected",
                "confidence": 0.0,
                "crop_image_path": "",
                "source_image": file_name
            }
            confidences["error"] = 0.0

        # Save results
        result = {"fields": fields, "confidences": confidences}
        save_results(result, annotated, output_dir, file_name)
        return result

    except Exception as e:
        logger.error(f"Selfie processing failed: {str(e)}")
        fields["error"] = {
            "ocr_text": f"Processing failed: {str(e)}",
            "confidence": 0.0,
            "crop_image_path": "",
            "source_image": file_name
        }
        confidences["error"] = 0.0
        result = {"fields": fields, "confidences": confidences}
        save_results(result, annotated, output_dir, file_name)
        return result