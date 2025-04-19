import os
import cv2
import json
import logging
import numpy as np
from ultralytics import YOLO
import pytesseract
import easyocr
from fuzzywuzzy import fuzz, process
import hashlib
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load YOLOv8 and EasyOCR
model = YOLO("yolo_models/citizenship.pt")
easyocr_reader = easyocr.Reader(['ne'], gpu=False)  # Prioritize Nepali
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Target fields for Nepali Citizenship documents
KNOWN_FIELDS = [
    "Name", "Father's Name", "Mother's Name", "Date of Birth",
    "Place of Birth", "Permanent Address", "Temporary Address",
    "Citizenship Number", "Issued District", "Gender",
    "Spouse Name", "Grandfather's Name", "Issued Date"
]

# Nepali + English hints for citizenship field recognition
NEPALI_HINTS = {
    "नाम": "Name", "नाउँ": "Name",
    "बुबाको नाम": "Father's Name", "पिताको नाम": "Father's Name",
    "आमाको नाम": "Mother's Name", "बाजेको नाम": "Grandfather's Name",
    "पति": "Spouse Name", "पत्नी": "Spouse Name",
    "लिङ्ग": "Gender", "पुरुष": "Gender", "महिला": "Gender",
    "स्थायी": "Permanent Address", "अस्थायी": "Temporary Address",
    "ठेगाना": "Permanent Address", "जन्म": "Date of Birth",
    "जन्म स्थान": "Place of Birth", "मिति": "Date of Birth",
    "ना.प्र.नं": "Citizenship Number", "ना.प्र.सं": "Citizenship Number",
    "जारी जिल्ला": "Issued District", "जारी मिति": "Issued Date"
}

# Clean OCR text
def clean_text(text):
    text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in "-:/")
    return text.strip()

# Smart field classification from OCR
def get_best_field_match(text):
    if not text.strip():
        return "Unknown"
    for hint, field in NEPALI_HINTS.items():
        if hint.lower() in text.lower():
            return field
    words = text.lower().split()
    for word in words:
        for hint, field in NEPALI_HINTS.items():
            if word in hint.lower():
                return field
    best, score = process.extractOne(text.lower(), KNOWN_FIELDS, scorer=fuzz.partial_ratio)
    return best if score > 40 else "Unknown"  # Lowered threshold

# OCR with fallback: EasyOCR (ne) → Tesseract (nep-fuse-2, nep)
def extract_text(image_crop):
    # Preprocess: contrast enhancement and binarization
    if len(image_crop.shape) == 2 or image_crop.shape[2] == 1:  # Grayscale
        gray = image_crop
    else:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try EasyOCR with Nepali
    try:
        results = easyocr_reader.readtext(binary, allowlist=None, language='ne')
        text = clean_text(" ".join(res[1] for res in results))
        if len(text) >= 3:
            return text, "easyocr"
    except:
        pass

    # Fallback to Tesseract (nep-fuse-2 first)
    for lang in ["nep-fuse-2", "nep"]:
        try:
            raw = pytesseract.image_to_string(binary, lang=lang).strip()
            text = clean_text(raw)
            if len(text) >= 3:
                return text, lang
        except:
            continue

    return "", "none"

# Crop enhancer
def enhance_crop(crop):
    # Handle grayscale or BGR input
    if len(crop.shape) == 2 or crop.shape[2] == 1:  # Grayscale
        gray = crop
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
    return sharpened

# Check for better label on rotations
def try_rotations_for_best_label(crop):
    angles = [0, 90, 180, 270]
    best_field, best_text, best_crop = "Unknown", "", crop
    best_conf = 0.0
    for angle in angles:
        rotated_crop = cv2.rotate(crop, {
            0: None,
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }[angle]) if angle != 0 else crop
        text, source = extract_text(rotated_crop)
        field = get_best_field_match(text)
        conf = len(text) / 100.0 if text and len(text) >= 3 else 0.0
        if field != "Unknown" and conf > best_conf:
            best_field, best_text, best_crop, best_conf = field, text, rotated_crop, conf
    return best_field, best_text, best_crop

# Generate crop hash to avoid duplicates
def get_crop_hash(crop: np.ndarray) -> str:
    crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)
    if len(crop.shape) == 2 or crop.shape[2] == 1:
        gray = crop
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, buffer = cv2.imencode('.png', binary)
    return hashlib.md5(buffer).hexdigest()

# Global variables
seen_hashes = set()

# Process individual image
def process_image(img_path: str, output_dir: str) -> Dict[str, Any]:
    global seen_hashes
    seen_hashes.clear()  # Reset per run
    all_results = {}
    annotated_final = None
    best_image_path = None
    max_fields = 0
    total_confidence = 0.0

    # Load original image
    original_image = cv2.imread(img_path)
    if original_image is None:
        logging.error(f"Failed to load image: {img_path}")
        return {
            "fields": {"error": {"ocr_text": "Failed to load image"}},
            "confidences": {"error": 0.0},
            "annotated_path": ""
        }

    file_name = os.path.basename(img_path)
    crop_dir = os.path.join(output_dir, "crops")
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)

    # List of images to process: original + all preprocessed
    image_paths = [(img_path, 1.0)]  # (path, score)
    if os.path.exists(preprocess_dir):
        for fname in os.listdir(preprocess_dir):
            if fname.endswith(('.jpg', '.png')) and not fname.startswith('face_'):
                score = 0.5
                if 'binarized' in fname or 'bw' in fname:
                    score = 0.95
                elif 'rotated_90' in fname:
                    score = 0.9
                elif 'rotated_180' in fname:
                    score = 0.8
                elif 'rotated_270' in fname:
                    score = 0.7
                image_paths.append((os.path.join(preprocess_dir, fname), score))

    # Sort by score but process all
    image_paths.sort(key=lambda x: x[1], reverse=True)

    for img_path, _ in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Failed to load preprocessed image: {img_path}")
            continue

        # Try rotations on full image
        for angle in [0, 90, 180, 270]:
            rotated_image = image.copy()
            if angle != 0:
                rotated_image = cv2.rotate(rotated_image, {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                }[angle])

            try:
                results = model.predict(rotated_image, conf=0.4, max_det=40, device="cpu", verbose=False)
            except Exception as e:
                logging.error(f"YOLO error at {img_path}, angle {angle}: {str(e)}")
                continue

            annotated = rotated_image.copy()
            temp_best = {}

            # Spatial deduplication
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            keep = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                area_i = (x2 - x1) * (y2 - y1)
                overlap = False
                for j in keep:
                    x1_j, y1_j, x2_j, y2_j = map(int, boxes[j])
                    xi1, yi1 = max(x1, x1_j), max(y1, y1_j)
                    xi2, yi2 = min(x2, x2_j), min(y2, y2_j)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    if inter_area / area_i > 0.6:
                        overlap = True
                        if confs[i] > confs[j]:
                            keep[keep.index(j)] = i
                        break
                if not overlap:
                    keep.append(i)

            for i in keep:
                x1, y1, x2, y2 = map(int, boxes[i])
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(rotated_image.shape[1], x2 + pad_x), min(rotated_image.shape[0], y2 + pad_y)

                # Crop for OCR
                crop = rotated_image[y1:y2, x1:x2]
                try:
                    crop = enhance_crop(crop)
                except Exception as e:
                    logging.error(f"Enhance crop failed at {img_path}, angle {angle}: {str(e)}")
                    continue

                crop_hash = get_crop_hash(crop)
                if crop_hash in seen_hashes:
                    continue
                seen_hashes.add(crop_hash)

                field, text, crop = try_rotations_for_best_label(crop)
                if field == "Unknown":
                    continue

                conf = float(confs[i])
                crop_name = f"{os.path.splitext(file_name)[0]}_rotated_{angle}_{field}_{x1}_{y1}.jpg"
                crop_path = os.path.join(crop_dir, crop_name)
                cv2.imwrite(crop_path, crop)

                field_result = {
                    "ocr_text": text,
                    "confidence": round(conf, 3),
                    "crop_image_path": crop_path,
                    "source_image": file_name
                }

                if field in temp_best:
                    if conf > temp_best[field]["confidence"]:
                        temp_best[field] = field_result
                else:
                    temp_best[field] = field_result

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{field}: {text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Combine results
            for field, result in temp_best.items():
                if field not in all_results or result["confidence"] > all_results[field]["confidence"]:
                    all_results[field] = result

            # Update annotated image
            current_conf = sum(result["confidence"] for result in temp_best.values())
            if len(temp_best) > max_fields or (len(temp_best) == max_fields and current_conf > total_confidence):
                max_fields = len(temp_best)
                total_confidence = current_conf
                annotated_final = annotated.copy()
                best_image_path = img_path

            # Early stopping
            if len(all_results) >= 0.8 * len(KNOWN_FIELDS) and all(result["confidence"] > 0.7 for result in all_results.values()):
                break
        else:
            continue
        break

    # Prepare fields and confidences
    confidences = {k: v["confidence"] for k, v in all_results.items()}
    if not all_results:
        all_results["error"] = {
            "ocr_text": "No fields detected",
            "confidence": 0.0,
            "crop_image_path": "",
            "source_image": file_name
        }
        confidences["error"] = 0.0

    # Fill gaps for undetected fields
    for field in KNOWN_FIELDS:
        if field not in all_results:
            all_results[field] = {
                "ocr_text": "",
                "confidence": 0.0,
                "crop_image_path": "",
                "source_image": file_name
            }
            confidences[field] = 0.0

    # Save annotated image
    annotated_path = os.path.join(output_dir, f"annotated_{file_name}")
    if annotated_final is not None:
        cv2.imwrite(annotated_path, annotated_final)
    else:
        cv2.imwrite(annotated_path, original_image)

    # Save fields JSON
    with open(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_fields.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # Save final_combined.json
    fields = {k: v["ocr_text"] for k, v in all_results.items() if k != "error"}
    confidences_json = {k: v["confidence"] for k, v in all_results.items() if k != "error"}
    if "error" in all_results:
        fields["error"] = all_results["error"]["ocr_text"]
        confidences_json["error"] = all_results["error"]["confidence"]

    final_json_path = os.path.join(output_dir, "final_combined.json")
    combined_data = []
    if os.path.exists(final_json_path):
        try:
            with open(final_json_path, "r", encoding="utf-8") as f:
                combined_data = json.load(f)
            if not isinstance(combined_data, list):
                logging.warning(f"final_combined.json is not a list, resetting: {final_json_path}")
                combined_data = []
        except Exception as e:
            logging.error(f"Failed to read final_combined.json: {str(e)}")
            combined_data = []

    run_id = os.path.basename(output_dir)
    existing_entry = next((entry for entry in combined_data if entry["run_id"] == run_id), None)
    new_entry = {
        "run_id": run_id,
        "document_type": "Citizenship",
        "timestamp": datetime.now().isoformat(),
        "fields": fields,
        "confidences": confidences_json
    }

    if existing_entry:
        combined_data[combined_data.index(existing_entry)] = new_entry
    else:
        combined_data.append(new_entry)

    try:
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved final_combined.json: {final_json_path}")
    except Exception as e:
        logging.error(f"Failed to save final_combined.json: {str(e)}")

    logging.info(f"Saved results for {file_name}: {annotated_path}")
    return {
        "fields": all_results,
        "confidences": confidences,
        "annotated_path": annotated_path
    }