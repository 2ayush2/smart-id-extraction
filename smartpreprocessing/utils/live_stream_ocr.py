import cv2
import numpy as np
import pytesseract
import easyocr
import logging
from utils.inference import YOLOv10
from utils.face_verify import verify_faces

# Configure logging (consistent with app.py)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_id_extraction.log")]
)
logger = logging.getLogger(__name__)

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLOv10 model from Hugging Face
from huggingface_hub import hf_hub_download
try:
    model_path = hf_hub_download("onnx-community/yolov10n", "onnx/model.onnx")
    yolo = YOLOv10(model_path)
    logger.debug("YOLOv10 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLOv10 model: {str(e)}")
    yolo = None

# Initialize EasyOCR for English and Nepali
ocr_reader = easyocr.Reader(['en', 'ne'], gpu=False)

def extract_text(crop):
    """
    Extract text from a cropped image region using Tesseract and EasyOCR.
    
    Args:
        crop: Cropped image (numpy array).
    
    Returns:
        Extracted text (str).
    """
    try:
        # Try Tesseract with Nepali (lang="nep-fuse-2" as specified)
        text = pytesseract.image_to_string(crop, lang="nep-fuse-2").strip()
        if len(text) >= 3:
            logger.debug(f"Extracted text with Tesseract: {text}")
            return text
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {str(e)}")
    
    try:
        # Fallback to EasyOCR
        results = ocr_reader.readtext(crop, detail=0)
        text = " ".join(results).strip()
        logger.debug(f"Extracted text with EasyOCR: {text}")
        return text
    except Exception as e:
        logger.error(f"EasyOCR failed: {str(e)}")
        return ""

def process_frame(frame, threshold=0.3, doc_type="License"):
    """
    Process a single video frame for document detection and text extraction.
    
    Args:
        frame: Video frame (numpy array).
        threshold: Confidence threshold for YOLOv10 detections.
        doc_type: Document type ("Citizenship", "License", "Passport").
    
    Returns:
        Tuple of (output dictionary, annotated frame).
    """
    if yolo is None:
        logger.error("YOLOv10 model not loaded")
        return {"error": "Model not loaded"}, frame

    output = {
        "document_type": doc_type,
        "extracted_fields": {},
        "is_verified": False,
        "face_score": 0.0
    }

    # Define field labels based on document type
    if doc_type == "Citizenship":
        field_labels = {
            0: "Name",
            1: "Date of Birth",
            2: "Address",
            3: "Citizenship Number",
            4: "Father Name",
            5: "Mother Name"
        }
    elif doc_type == "Passport":
        field_labels = {
            0: "Passport Number",
            1: "Name",
            2: "Nationality",
            3: "Date of Birth",
            4: "Sex",
            5: "Date of Expiry"
        }
    else:  # License
        field_labels = {
            0: "Name",
            1: "DOB",
            2: "Address",
            3: "License Number",
            4: "Issued Date",
            5: "Expiry Date"
        }

    annotated = frame.copy()
    # Resize frame to YOLOv10 input size
    image = cv2.resize(frame, (yolo.input_width, yolo.input_height))
    img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    try:
        outputs = yolo.session.run(None, {yolo.input_name: img_input})[0][0]
    except Exception as e:
        logger.error(f"YOLOv10 inference failed: {str(e)}")
        output["error"] = "Inference failed"
        return output, annotated

    for obj in outputs:
        conf = obj[4]
        if conf < threshold:
            continue
        x, y, w, h = obj[:4]
        cls_id = int(obj[5]) if len(obj) > 5 else -1
        label = field_labels.get(cls_id, f"Field {cls_id}")
        x1 = int((x - w / 2) * annotated.shape[1])
        y1 = int((y - h / 2) * annotated.shape[0])
        x2 = int((x + w / 2) * annotated.shape[1])
        y2 = int((y + h / 2) * annotated.shape[0])

        # Fix bounds calculation (corrected syntax error)
        x1, y1, x2, y2 = map(lambda v: max(0, min(v, annotated.shape[1] if v % 2 == 0 else annotated.shape[0]-1)), [x1, y1, x2, y2])
        crop = annotated[y1:y2, x1:x2]
        text = extract_text(crop)

        if text:
            output['extracted_fields'][label] = {
                "text": text,
                "confidence": round(conf, 3)
            }

        # Draw bounding box and label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Perform face verification
    try:
        is_verified, face_score = verify_faces(annotated, annotated)
        output["is_verified"] = is_verified
        output["face_score"] = round(face_score, 4)
        logger.debug(f"Face verification result - is_verified: {is_verified}, face_score: {face_score}")
    except Exception as e:
        logger.error(f"Face verification failed: {str(e)}")

    return output, annotated