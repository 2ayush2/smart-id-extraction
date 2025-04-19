import os
import json
import cv2
import pytesseract
import string as st
from passporteye import read_mrz
from dateutil import parser
import easyocr
import logging
import numpy as np  # Added missing import
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Load country codes
try:
    with open('utils/country_codess.json', 'r', encoding='utf-8') as f:
        country_codes = json.load(f)
    logger.info("Country codes loaded successfully")
except Exception as e:
    logger.error(f"Failed to load country codes: {str(e)}")
    country_codes = []

# Initialize EasyOCR (CPU)
try:
    reader = easyocr.Reader(['en'], gpu=False)
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {str(e)}")
    reader = None

# === Utilities ===

def parse_date(text: str) -> str:
    """Parse a date string and return it in DD/MM/YYYY format."""
    try:
        return parser.parse(text, yearfirst=True).strftime('%d/%m/%Y')
    except:
        logger.warning(f"Invalid date format: {text}")
        return "Invalid Date"

def clean(text: str) -> str:
    """Clean text by keeping only alphanumeric characters and converting to uppercase."""
    return ''.join(i for i in text if i.isalnum()).upper()

def get_country_name(code: str) -> str:
    """Convert a country code to its full name."""
    code = code.upper()
    for c in country_codes:
        if c['alpha-3'] == code:
            return c['name'].upper()
    logger.warning(f"Country code not found: {code}")
    return code

def get_sex(code: str) -> str:
    """Determine the sex from a code."""
    code = code.upper()
    if code in ['M', 'F']:
        return code
    if code == '0':
        return 'M'
    logger.warning(f"Unknown sex code: {code}")
    return 'UNKNOWN'

# === MRZ OCR Engines ===

def extract_with_tesseract(gray: np.ndarray) -> list:
    """Extract text from an image using Tesseract OCR."""
    if gray.dtype != 'uint8':
        gray = (gray * 255).astype('uint8')
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    try:
        lines = pytesseract.image_to_string(gray, config=config).splitlines()
        return [line.strip().upper() for line in lines if line.strip()]
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {str(e)}")
        return []

def extract_with_easyocr(gray: np.ndarray) -> list:
    """Extract text from an image using EasyOCR."""
    if not reader:
        logger.error("EasyOCR not available")
        return []
    if gray.dtype != 'uint8':
        gray = (gray * 255).astype('uint8')
    allowed_chars = st.ascii_letters + st.digits + '< '
    try:
        result = reader.readtext(gray, paragraph=False, detail=0, allowlist=allowed_chars)
        return [line.strip().upper() for line in result if line.strip()]
    except Exception as e:
        logger.error(f"EasyOCR failed: {str(e)}")
        return []

# === MRZ Parser ===

def parse_mrz(lines: list) -> tuple[Optional[dict], float]:
    """Parse MRZ lines and extract passport data."""
    if len(lines) < 2:
        logger.warning("Insufficient MRZ lines")
        return None, 0

    a = lines[0].ljust(44, '<')[:44]
    b = lines[1].ljust(44, '<')[:44]
    try:
        surname, names = a[5:44].split('<<', 1) if '<<' in a[5:44] else (a[5:44], '')
        data = {
            'Name': names.replace('<', ' ').strip(),
            'Surname': surname.replace('<', ' ').strip(),
            'Sex': get_sex(clean(b[20])),
            'Date of Birth': parse_date(b[13:19]),
            'Nationality': get_country_name(clean(b[10:13])),
            'Passport Type': clean(a[0:2]),
            'Passport Number': clean(b[0:9]),
            'Issuing Country': get_country_name(clean(a[2:5])),
            'Expiration Date': parse_date(b[21:27]),
            'Personal Number': clean(b[28:42])
        }

        # Weighted field scoring for accuracy
        weights = {
            'Passport Number': 2,
            'Name': 1.5,
            'Surname': 1.5,
            'Date of Birth': 2,
            'Expiration Date': 1.5,
            'Nationality': 1,
            'Issuing Country': 1,
            'Sex': 0.5,
            'Passport Type': 0.5,
            'Personal Number': 0.5
        }
        score = sum(weights[k] for k, v in data.items() if v and v != "UNKNOWN" and "Invalid" not in v)
        logger.debug(f"MRZ parsed successfully, score: {score}")
        return data, score
    except Exception as e:
        logger.error(f"MRZ parsing failed: {str(e)}")
        return None, 0

# === MRZ Evaluation ===

def evaluate_image(img_path: str, cancel_flag: bool = False) -> Optional[dict]:
    """Evaluate an image for MRZ data using multiple OCR methods."""
    if cancel_flag:
        logger.info("Processing cancelled during image evaluation")
        return None

    logger.info(f"Checking MRZ in: {img_path}")
    try:
        mrz = read_mrz(img_path, save_roi=True)
        if not mrz:
            logger.warning("No MRZ found")
            return None

        img = mrz.aux['roi']
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Tesseract
        if cancel_flag:
            logger.info("Processing cancelled during Tesseract OCR")
            return None

        tess_lines = extract_with_tesseract(gray)
        data_tess, score_tess = parse_mrz(tess_lines)

        if score_tess >= 10:
            logger.info(f"Tesseract score {score_tess}. Early accept.")
            return {'data': data_tess, 'method': 'Tesseract', 'score': score_tess, 'image': img_path}

        # EasyOCR
        if cancel_flag:
            logger.info("Processing cancelled during EasyOCR")
            return None

        easy_lines = extract_with_easyocr(gray)
        data_easy, score_easy = parse_mrz(easy_lines)

        logger.info(f"Tesseract Score: {score_tess}, EasyOCR Score: {score_easy}")

        if score_easy > score_tess:
            return {'data': data_easy, 'method': 'EasyOCR', 'score': score_easy, 'image': img_path}
        elif data_tess:
            return {'data': data_tess, 'method': 'Tesseract', 'score': score_tess, 'image': img_path}
        return None

    except Exception as e:
        logger.error(f"Failed to evaluate image {img_path}: {str(e)}")
        return None

# === Main Processing Function ===

def process_image(img_path: str, output_dir: str, cancel_flag: bool = False) -> Dict[str, Any]:
    """
    Process a passport image to extract MRZ data.
    Args:
        img_path: Path to the input image.
        output_dir: Directory to save outputs (e.g., annotated image, JSON).
        cancel_flag: Flag to check for cancellation.
    Returns:
        Dictionary with fields, confidences, and score.
    """
    try:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            return {
                "fields": {"error": {"ocr_text": "Failed to load image", "confidence": 0.0}},
                "confidences": {"error": 0.0},
                "score": 0
            }

        # Evaluate image for MRZ data
        result = evaluate_image(img_path, cancel_flag)
        if cancel_flag:
            logger.info("Processing cancelled during evaluation")
            return {
                "fields": {"error": {"ocr_text": "Processing cancelled", "confidence": 0.0}},
                "confidences": {"error": 0.0},
                "score": 0
            }

        if not result:
            logger.warning("No valid MRZ extracted")
            return {
                "fields": {"error": {"ocr_text": "No valid MRZ extracted", "confidence": 0.0}},
                "confidences": {"error": 0.0},
                "score": 0
            }

        # Prepare fields and confidences for the pipeline
        data = result['data']
        score = result['score']
        method = result['method']
        fields = {}
        confidences = {}
        confidence = 0.9 if method == 'Tesseract' else 0.85  # Assign confidence based on method

        for field, value in data.items():
            if value and value != "UNKNOWN" and "Invalid" not in value:
                fields[field] = {
                    "ocr_text": value,
                    "confidence": confidence,
                    "crop_image_path": "",  # No cropping in this implementation
                    "source_image": os.path.basename(img_path)
                }
                confidences[field] = confidence
            else:
                logger.debug(f"Skipping field {field} with value {value}")

        if not fields:
            logger.warning("No valid fields extracted after filtering")
            return {
                "fields": {"error": {"ocr_text": "No valid fields extracted", "confidence": 0.0}},
                "confidences": {"error": 0.0},
                "score": score
            }

        # Save annotated image (just a copy of the original for now)
        annotated_filename = f"annotated_{os.path.basename(img_path)}"
        annotated_path = os.path.join(output_dir, annotated_filename).replace("\\", "/")
        cv2.imwrite(annotated_path, image)
        logger.debug(f"Saved annotated image: {annotated_path}")

        logger.info(f"Successfully processed passport image: {img_path}, score: {score}")
        return {
            "fields": fields,
            "confidences": confidences,
            "score": score
        }

    except Exception as e:
        logger.error(f"Processing failed for {img_path}: {str(e)}")
        return {
            "fields": {"error": {"ocr_text": str(e), "confidence": 0.0}},
            "confidences": {"error": 0.0},
            "score": 0
        }