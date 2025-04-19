import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import logging
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FaceAnalysis
try:
    FACE_APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    FACE_APP.prepare(ctx_id=0, det_size=(320, 320))  # Reduced det_size for speed
    logger.info("FaceAnalysis initialized")
except Exception as e:
    logger.warning(f"Failed to initialize FaceAnalysis: {str(e)}. Using fallback.")
    FACE_APP = None

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    try:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    except Exception as e:
        logger.error(f"Failed to rotate image: {str(e)}")
        return image

def enhance_image(image: np.ndarray) -> np.ndarray:
    try:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)  # Reduced window size
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        smooth = cv2.bilateralFilter(enhanced, 5, 75, 75)
        gaussian = cv2.GaussianBlur(smooth, (0, 0), 2.0)
        sharpened = cv2.addWeighted(smooth, 1.8, gaussian, -0.8, 0)
        return sharpened
    except Exception as e:
        logger.warning(f"Image enhancement failed: {str(e)}")
        return image

def crop_face(image: np.ndarray, output_dir: str, save_name: str) -> Optional[str]:
    if FACE_APP is None:
        logger.warning("FaceAnalysis not initialized. Cannot crop face.")
        return None
    try:
        faces = FACE_APP.get(image)
        if not faces:
            logger.warning("No faces detected for cropping.")
            return None
        face = faces[0]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        padding = int((x2 - x1) * 0.2)
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        cropped_face = image[y1:y2, x1:x2]
        crops_dir = os.path.join(output_dir, "crops").replace("\\", "/")
        os.makedirs(crops_dir, exist_ok=True)
        cropped_face_path = os.path.join(crops_dir, f"{save_name}_face.jpg").replace("\\", "/")
        cv2.imwrite(cropped_face_path, cropped_face)
        logger.debug(f"Saved cropped face image: {cropped_face_path}")
        return cropped_face_path
    except Exception as e:
        logger.warning(f"Face cropping failed: {str(e)}")
        return None

def save_image(args: Tuple[np.ndarray, str, str]):
    """Helper function to save image in parallel."""
    image, path, log_message = args
    try:
        cv2.imwrite(path, image)
        logger.debug(log_message)
    except Exception as e:
        logger.error(f"Failed to save image {path}: {str(e)}")

def preprocess_image(image_path: str, output_dir: str, output_suffix: str = "") -> Tuple[Optional[str], Optional[str], Optional[str], List[Dict[str, Any]]]:
    try:
        # Create directories once
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot load image from {image_path}")
            return None, None, None, []
        if image.shape[0] < 100 or image.shape[1] < 100:
            logger.warning(f"Image too small: {image_path}")
            return None, None, None, []
        logger.debug(f"Image loaded, shape: {image.shape}")

        base_name = os.path.splitext(os.path.basename(image_path))[0] if not output_suffix else output_suffix
        angles = [0, 90, 180, 270]
        preproc_images = []

        # Enhance once
        enhanced = enhance_image(image)

        # Process rotations
        save_tasks = []
        for angle in angles:
            rotated = enhanced if angle == 0 else rotate_image(enhanced, angle)  # Skip rotation for 0°
            rotated_path = os.path.join(output_dir, f"{base_name}_rotated_{angle}.jpg").replace("\\", "/")
            save_tasks.append((rotated, rotated_path, f"Saved rotated and enhanced image: {rotated_path}"))
            preproc_images.append({"path": rotated_path, "type": f"rotated_{angle}"})

        # Crop face from original image
        cropped_face_path = crop_face(image, output_dir, base_name)

        # Process for binarized and BW versions
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Binarized
        binarized = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3  # Smaller block size
        )
        binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        binarized_path = os.path.join(output_dir, f"{base_name}_binarized.jpg").replace("\\", "/")
        save_tasks.append((binarized, binarized_path, f"Saved binarized image: {binarized_path}"))
        preproc_images.append({"path": binarized_path, "type": "binarized"})

        # Black-and-white
        _, bw_output = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        bw_path = os.path.join(output_dir, f"{base_name}_bw.jpg").replace("\\", "/")
        save_tasks.append((bw_output, bw_path, f"Saved black-and-white image: {bw_path}"))
        preproc_images.append({"path": bw_path, "type": "bw"})

        # Save images in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(save_image, save_tasks)

        return binarized_path, save_tasks[0][1], cropped_face_path, preproc_images
    except Exception as e:
        logger.error(f"Error in preprocess_image for {image_path}: {str(e)}")
        return None, None, None, []

def preprocess_selfie(image_path: str, output_dir: str, output_suffix: str = "selfie") -> Tuple[Optional[str], Optional[str], Optional[str], List[Dict[str, Any]]]:
    try:
        # Create directories once
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot load selfie from {image_path}")
            return None, None, None, []
        if image.shape[0] < 100 or image.shape[1] < 100:
            logger.warning(f"Selfie too small: {image_path}")
            return None, None, None, []
        logger.debug(f"Selfie loaded, shape: {image.shape}")

        base_name = os.path.splitext(os.path.basename(image_path))[0] if not output_suffix else output_suffix
        preproc_images = []

        # Enhance image
        enhanced = enhance_image(image)
        best_img_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg").replace("\\", "/")
        save_tasks = [(enhanced, best_img_path, f"Saved enhanced selfie: {best_img_path}")]
        preproc_images.append({"path": best_img_path, "type": "enhanced"})

        # Crop face
        face_path = crop_face(image, output_dir, base_name)
        logger.debug(f"Selfie cropped face path: {face_path if face_path else 'None'}")

        # Generate binarized and black-and-white versions
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
        )
        binarized_path = os.path.join(output_dir, f"{base_name}_binarized.jpg").replace("\\", "/")
        save_tasks.append((binarized, binarized_path, f"Saved binarized selfie: {binarized_path}"))
        preproc_images.append({"path": binarized_path, "type": "binarized"})

        _, bw_output = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        bw_path = os.path.join(output_dir, f"{base_name}_bw.jpg").replace("\\", "/")
        save_tasks.append((bw_output, bw_path, f"Saved black-and-white selfie: {bw_path}"))
        preproc_images.append({"path": bw_path, "type": "bw"})

        # Save images in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(save_image, save_tasks)

        # Check if enhanced image was saved
        if not os.path.exists(best_img_path):
            logger.error(f"Enhanced selfie not saved: {best_img_path}")
            return None, None, None, []

        return binarized_path, best_img_path, face_path, preproc_images
    except Exception as e:
        logger.error(f"Error in preprocess_selfie for {image_path}: {str(e)}")
        return None, None, None, []