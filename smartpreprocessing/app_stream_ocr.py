import gradio as gr
from gradio_webrtc import WebRTC
import cv2
import os
import numpy as np
from huggingface_hub import hf_hub_download
from utils.inference import YOLOv10

from utils.face_verify import verify_faces
import easyocr
import pytesseract

# Initialize EasyOCR
ocr_reader = easyocr.Reader(['en', 'ne'], gpu=False)

# Load YOLOv10 model
model_path = hf_hub_download("onnx-community/yolov10n", "onnx/model.onnx")
yolo = YOLOv10(model_path)

def extract_text(crop):
    try:
        # Try Tesseract Nepali first
        nep_text = pytesseract.image_to_string(crop, lang="nep-fuse-2").strip()
        if len(nep_text) >= 3:
            return nep_text
    except Exception as e:
        print(f"Nepali OCR failed: {e}")

    try:
        # Fallback to EasyOCR
        results = ocr_reader.readtext(crop, detail=0)
        easy_text = " ".join(results).strip()
        if len(easy_text) >= 3:
            return easy_text
    except Exception as e:
        print(f"EasyOCR failed: {e}")

    return ""

def detect_and_extract(image, threshold=0.3, doc_type="License"):
    output = {
        "document_type": doc_type,
        "extracted_fields": {},
        "is_verified": False,
        "face_score": 0.0
    }

    # Set dynamic field labels based on doc_type
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

    annotated = image.copy()
    detected_fields = {}
    image = cv2.resize(image, (yolo.input_width, yolo.input_height))
    img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    outputs = yolo.session.run(None, {yolo.input_name: img_input})[0][0]
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

        x1, y1, x2, y2 = map(lambda v: max(0, min(v, annotated.shape[1 if v%2==0 else 0]-1)), [x1, y1, x2, y2])
        crop = annotated[y1:y2, x1:x2]
        text = extract_text(crop)

        if text:
            detected_fields[label] = {
                "text": text,
                "confidence": round(conf, 3)
            }

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Face verification using the same frame as webcam input
    is_verified, face_score = verify_faces(annotated, annotated)

    output["extracted_fields"] = detected_fields
    output["is_verified"] = is_verified
    output["face_score"] = round(face_score, 4)

    return output, annotated

# RTC config (STUN server only for local/dev)
rtc_configuration = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"}
    ]
}

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Smart ID Real-Time OCR + Face Verification (YOLOv10 + WebRTC)")

    with gr.Row():
        doc_type = gr.Dropdown(["Citizenship", "License", "Passport"], label="Document Type", value="License")
        webrtc = WebRTC(rtc_configuration=rtc_configuration, label="📷 Live Stream")
        threshold = gr.Slider(0.2, 1.0, value=0.3, label="Confidence Threshold")

    with gr.Row():
        output_json = gr.JSON(label="📄 Extracted Info")
        annotated_img = gr.Image(label="🖼️ Annotated Output")

    webrtc.stream(fn=detect_and_extract, inputs=[webrtc, threshold, doc_type], outputs=[output_json, annotated_img], time_limit=10)

if __name__ == "__main__":
    demo.launch()