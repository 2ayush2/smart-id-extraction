import os
import cv2
import json
import re
import hashlib
import numpy as np
import easyocr
from ultralytics import YOLO
from typing import Dict, Any, List, Tuple

# Initialize models (load only once)
YOLO_MODEL = YOLO("yolo_models/lisc.pt")
EASY_OCR = easyocr.Reader(['en'], gpu=False)

# Field definitions
KNOWN_FIELDS = [
    "License Number", "Blood Group", "Name", "Address", "License Office",
    "Date of Birth", "Father/Husband Name", "Citizenship Number",
    "Category", "Passport Number", "Contact Number",
    "Issued Date", "Expiry Date"
]

LABEL_MAPPING = {
    "D.L.No": "License Number", "DL No": "License Number", "DLNo": "License Number",
    "B.G.": "Blood Group", "Blood Group": "Blood Group", "Name": "Name",
    "Address": "Address", "License Office": "License Office", "D.O.B.": "Date of Birth",
    "DOB": "Date of Birth", "F/H Name": "Father/Husband Name", "Father Name": "Father/Husband Name",
    "Citizenship No": "Citizenship Number", "Category": "Category",
    "Passport No": "Passport Number", "Contact No": "Contact Number",
    "D.O.I.": "Issued Date", "D.O.E.": "Expiry Date"
}


LABEL_KEYWORDS = {k.lower(): v for k, v in LABEL_MAPPING.items()}

def clean_text(text: str) -> str:
    """Clean text by removing unwanted characters and normalizing spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\-:/.,]", "", text)).strip()

def enhance_crop(crop: np.ndarray) -> np.ndarray:
    """Enhance crop for better OCR (optimized for speed)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Fast resize and sharpen
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def get_crop_hash(crop: np.ndarray) -> str:
    """Generate hash for crop to avoid duplicates."""
    _, buffer = cv2.imencode('.png', crop)
    return hashlib.md5(buffer).hexdigest()

def extract_text(crop: np.ndarray) -> str:
    """Extract text from crop using EasyOCR with optimized settings."""
    try:
        results = EASY_OCR.readtext(crop, detail=0, paragraph=False, contrast_ths=0.1, adjust_contrast=0.5)
        return clean_text(" ".join(results))
    except:
        return ""

def infer_label_from_pattern(text: str) -> str:
    """Infer label based on text pattern, avoiding 'Unknown'."""
    text_clean = clean_text(text)
    text_lower = text_clean.lower()

    # Date-like patterns
    if re.match(r"^\d{2}[-/]\d{2}[-/]\d{4}$|^\d{4}[-/]\d{2}[-/]\d{2}$", text_clean):
        if "birth" in text_lower or "dob" in text_lower:
            return "Date of Birth"
        elif "issue" in text_lower or "doi" in text_lower:
            return "Issued Date"
        elif "expiry" in text_lower or "doe" in text_lower:
            return "Expiry Date"
        return "Date of Birth"  # Default for date-like patterns

    # License Number or Citizenship Number
    if re.match(r"^\d{2}[- ]?\d{2}[- ]?\d{6,}$", text_clean):
        return "License Number"
    elif re.match(r"^[\d\s\-]+$", text_clean):
        cleaned_num = text_clean.replace(" ", "").replace("-", "")
        return "Citizenship Number" if len(cleaned_num) > 8 else "License Number"

    # Contact Number
    elif re.match(r"^\d{10,}$", text_clean):
        return "Contact Number"

    # Text-based fields
    elif re.match(r"^[A-Za-z\s]+$", text_clean):
        if len(text_clean.split()) > 1:
            return "Name" if "name" in text_lower else "Address"
        return "Category" if text_clean.isupper() else "License Office"

    # Reassign fragments that might belong to License Number
    if re.search(r"\d{4,}", text_clean):
        return "License Number"

    return "Address"  # Fallback to a common field instead of "Unknown"

def match_label(text: str) -> List[Tuple[str, str]]:
    """Match labels in text and split into label-value pairs."""
    text_lower = text.lower()
    matches = []
    last_pos = 0

    # Find all keyword matches
    keyword_positions = []
    for label in LABEL_KEYWORDS:
        start_idx = text_lower.find(label.lower())
        if start_idx != -1:
            keyword_positions.append((start_idx, start_idx + len(label), LABEL_KEYWORDS[label.lower()]))

    # Sort by position
    keyword_positions.sort()
    for start, end, label in keyword_positions:
        if start > last_pos:
            segment = text[last_pos:start].strip()
            if segment:
                inferred_label = infer_label_from_pattern(segment)
                matches.append((inferred_label, segment))
        value = text[end:].strip(" :-")
        if value:
            matches.append((label, value))
        last_pos = end

    # Process remaining segment
    if last_pos < len(text):
        segment = text[last_pos:].strip()
        if segment:
            label = infer_label_from_pattern(segment)
            matches.append((label, segment))

    # If no matches, infer label for the entire text
    if not matches and text.strip():
        label = infer_label_from_pattern(text)
        matches.append((label, text))

    return matches

def postprocess_value(label: str, value: str) -> str:
    """Clean and format value based on label."""
    value = clean_text(value)

    if label == "License Number":
        match = re.search(r"\d{2}[- ]?\d{2}[- ]?\d{6,}", value)
        if match:
            number = match.group(0)
            return re.sub(r"\s", "", number)  # Keep hyphens, remove spaces
        # Fix common OCR errors (e.g., partial number like "81")
        digits = re.sub(r"[^\d]", "", value)
        if len(digits) >= 10:  # Minimum length for a valid License Number
            return f"{digits[:2]}-{digits[2:4]}-{digits[4:]}"
        return digits if digits else value

    elif label == "Citizenship Number":
        match = re.search(r"[\d\s\-]+", value)
        return re.sub(r"[^\d]", "", match.group(0)) if match else value

    elif label in ["Date of Birth", "Issued Date", "Expiry Date"]:
        match = re.search(r"\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2}", value)
        if match:
            date = match.group(0).replace("/", "-")
            parts = date.split("-")
            return f"{parts[2]}-{parts[1]}-{parts[0]}" if len(parts[0]) == 2 else date
        return value

    elif label == "Category":
        match = re.search(r"[A-Za-z0-9\s]+", value)
        return re.sub(r"\s", "", match.group(0)).upper() if match else value

    elif label == "Contact Number":
        match = re.search(r"\d+", value)
        return match.group(0) if match else value

    elif label in ["Name", "Father/Husband Name"]:
        return " ".join(word.capitalize() for word in value.split())

    elif label == "Address":
        return value.title()

    elif label == "License Office":
        return value.title()

    return value

def try_rotations_for_best_label(crop: np.ndarray) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Try rotations to find the best label-value pairs."""
    best_matches, best_crop = [], crop
    best_score = 0

    for angle in [0, 90, 180, 270]:
        rotated_crop = crop if angle == 0 else cv2.rotate(crop, {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }[angle])
        text = extract_text(rotated_crop)
        matches = match_label(text)

        # Score based on presence of high-priority fields
        score = sum(1 for label, _ in matches if label in ["License Number", "Date of Birth", "Name"])
        if score > best_score:
            best_score = score
            best_matches = matches
            best_crop = rotated_crop

    return best_matches, best_crop

def process_image(img_path: str, output_dir: str) -> Dict[str, Any]:
    """Process image to extract fields with YOLO and OCR."""
    seen_hashes = set()
    final_fields = {}
    confidences = {}

    # Load image
    original_image = cv2.imread(img_path)
    if original_image is None:
        return {
            "fields": {"error": {"ocr_text": "Failed to load image", "confidence": 0.0}},
            "confidences": {"error": 0.0},
            "annotated_path": ""
        }

    file_name = os.path.basename(img_path)
    crop_dir = os.path.join(output_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)

    best_annotated = None
    best_field_count = 0

    # Try different rotations
    for angle in [0, 90, 180, 270]:
        image = original_image if angle == 0 else cv2.rotate(original_image, {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }[angle])

        # YOLO prediction
        results = YOLO_MODEL.predict(image, conf=0.5, max_det=40, device="cpu", verbose=False)
        annotated = image.copy()
        current_fields = {}

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                pad_x, pad_y = int((x2 - x1) * 0.02), int((y2 - y1) * 0.02)
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)

                crop = image[y1:y2, x1:x2]
                crop = enhance_crop(crop)

                # Skip duplicates
                hash_key = get_crop_hash(crop)
                if hash_key in seen_hashes:
                    continue
                seen_hashes.add(hash_key)

                # Extract and match text
                matches, crop = try_rotations_for_best_label(crop)
                for label, raw_value in matches:
                    value = postprocess_value(label, raw_value)
                    crop_name = f"{os.path.splitext(file_name)[0]}_rotated_{angle}_{label}_{x1}_{y1}.jpg"
                    crop_path = os.path.join(crop_dir, crop_name)
                    cv2.imwrite(crop_path, crop)

                    field_result = {
                        "ocr_text": value,
                        "confidence": round(float(confs[i]), 3),
                        "crop_image_path": crop_path,
                        "source_image": file_name
                    }

                    # Update if confidence is higher
                    if label not in current_fields or field_result["confidence"] > current_fields[label]["confidence"]:
                        current_fields[label] = field_result

                    # Update global results
                    if label not in final_fields or field_result["confidence"] > final_fields[label]["confidence"]:
                        final_fields[label] = field_result
                        confidences[label] = field_result["confidence"]

                # Annotate image
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label}: {value}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update best annotated image
        if len(current_fields) > best_field_count:
            best_field_count = len(current_fields)
            best_annotated = annotated

    # Fill missing fields
    for field in KNOWN_FIELDS:
        if field not in final_fields:
            final_fields[field] = {
                "ocr_text": "",
                "confidence": 0.0,
                "crop_image_path": "",
                "source_image": file_name
            }
            confidences[field] = 0.0

    # Save annotated image
    annotated_path = os.path.join(output_dir, f"annotated_{file_name}")
    cv2.imwrite(annotated_path, best_annotated if best_annotated is not None else original_image)

    # Save fields to JSON
    with open(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_fields.json"), "w", encoding="utf-8") as f:
        json.dump(final_fields, f, indent=4, ensure_ascii=False)

    return {
        "fields": final_fields,
        "confidences": confidences,
        "annotated_path": annotated_path
    }