import os
import json
import tempfile
import shutil
import logging
import uuid
from datetime import datetime
from glob import glob
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from typing import List, Dict, Any, Optional
from PIL import Image
from pipeline.face_verifier import FaceVerifier
from pipeline.liveness_checker import LivenessChecker
from utils.preprocess import preprocess_image
from utils.citizenship_ocr import process_image as process_citizenship
from utils.license_ocr import process_image as process_license
from utils.passport_ocr import process_image as process_passport

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_id_extraction.log")]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart ID Extraction API")

# Initialize FaceVerifier and LivenessChecker
try:
    face_verifier = FaceVerifier()
    liveness_checker = LivenessChecker()
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    face_verifier = None
    liveness_checker = None

def combine_results(doc_type: str, output_folder: str) -> Dict[str, Any]:
    """
    Combine JSON results from multiple processed images, selecting the highest-confidence detections.
    
    Args:
        doc_type: Type of document ("Citizenship", "License", "Passport").
        output_folder: Directory containing JSON results.
    
    Returns:
        Dictionary with combined results.
    """
    try:
        json_files = glob(os.path.join(output_folder, "**/*.json"), recursive=True)
        if not json_files:
            raise ValueError("No JSON files found")
        
        fields = {}
        source_images = set()
        for json_path in json_files:
            if "final_combined" in os.path.basename(json_path):
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for det in data.get("detections", []):
                field = det["field_name"]
                confidence = det["confidence"]
                if field not in fields or confidence > fields[field]["confidence"]:
                    fields[field] = {
                        "value": det["ocr_text"],
                        "confidence": round(confidence, 3)
                    }
            source_images.add(data["file_name"])
        
        formatted_fields = {key: val["value"] for key, val in fields.items()}
        combined = {
            "document_type": doc_type,
            "fields": formatted_fields,
            "source_images": list(source_images)
        }
        
        combined_path = os.path.join(output_folder, "final_combined.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=4, ensure_ascii=False)
        logger.debug(f"Combined JSON saved: {combined_path}")
        return combined
    except Exception as e:
        logger.error(f"Error combining results: {str(e)}")
        return {"error": str(e)}

async def process_document(
    files: List[UploadFile],
    doc_type: str,
    selfie_file: Optional[UploadFile] = None
) -> Dict[str, Any]:
    """
    Process uploaded images for OCR, face verification, and liveness check.
    
    Args:
        files: List of ID image files.
        doc_type: Type of document ("Citizenship", "License", "Passport").
        selfie_file: Optional selfie image.
    
    Returns:
        Dictionary with results, processed files, and verification details.
    """
    run_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    preprocess_base_dir = os.path.join("preprocess", run_id)
    output_base_dir = os.path.join("ocr_result", run_id)
    uploaded_paths = []
    annotated_paths = []
    preprocessed_paths = []
    cropped_face_paths = []
    selfie_path = None
    
    try:
        if not files:
            raise ValueError("No ID images provided")
        if doc_type not in ["Citizenship", "License", "Passport"]:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        logger.debug(f"Processing {len(files)} images for {doc_type} (Run ID: {run_id})")
        os.makedirs(preprocess_base_dir, exist_ok=True)
        os.makedirs(output_base_dir, exist_ok=True)
        
        for idx, file in enumerate(files):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            input_name = f"input_{timestamp}_{idx}"
            input_path = os.path.join(temp_dir, f"{input_name}.jpg")
            
            with open(input_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_paths.append(input_path)
            logger.debug(f"Saved input image to {input_path}")
            
            preprocess_dir = os.path.join(preprocess_base_dir, input_name)
            output_dir = os.path.join(output_base_dir, input_name)
            os.makedirs(preprocess_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            if doc_type in ["Citizenship", "License"]:
                logger.debug(f"Preprocessing image {idx+1}")
                try:
                    _, final_path = preprocess_image(input_path, preprocess_dir, output_suffix="")
                    if not final_path or not os.path.exists(final_path):
                        logger.error(f"Preprocessing failed for {input_path}")
                        continue
                except Exception as e:
                    logger.error(f"Preprocessing error for {input_path}: {str(e)}")
                    continue
                
                all_preprocessed = glob(os.path.join(preprocess_dir, "[1-7]_*.jpg"))
                if not all_preprocessed:
                    logger.warning(f"No preprocessed images for {input_path}")
                    continue
                preprocessed_paths.extend(all_preprocessed)
                
                for img_path in all_preprocessed:
                    shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
                
                for preprocessed_path in all_preprocessed:
                    logger.debug(f"Running OCR on {preprocessed_path}")
                    json_result = None
                    if doc_type == "Citizenship":
                        json_result = process_citizenship(preprocessed_path, output_dir)
                    elif doc_type == "License":
                        json_result = process_license(preprocessed_path, output_dir)
                    
                    if json_result is None:
                        logger.warning(f"OCR failed for {preprocessed_path}")
                        continue
                    
                    file_name = os.path.basename(preprocessed_path)
                    annotated_path = os.path.join(output_dir, f"annotated_{file_name}")
                    if os.path.exists(annotated_path):
                        annotated_paths.append(annotated_path)
            
            elif doc_type == "Passport":
                original_path = os.path.join(output_dir, "input_image.jpg")
                shutil.copy(input_path, original_path)
                preprocessed_paths.append(original_path)
                
                logger.debug(f"Running OCR on original image {original_path}")
                json_result = process_passport(original_path, output_dir)
                
                if json_result is None:
                    logger.warning(f"OCR failed for {original_path}")
                    continue
                
                annotated_path = os.path.join(output_dir, "annotated_input_image.jpg")
                if os.path.exists(annotated_path):
                    annotated_paths.append(annotated_path)
        
        if selfie_file:
            selfie_path = os.path.join(temp_dir, "selfie.jpg")
            with open(selfie_path, "wb") as f:
                content = await selfie_file.read()
                f.write(content)
            logger.debug(f"Saved selfie image to {selfie_path}")
        
        combined_result = combine_results(doc_type, output_base_dir)
        
        face_verification_result = {
            "is_verified": False,
            "match_score": 0.0,
            "is_alive": False,
            "details": []
        }
        
        if not face_verifier:
            face_verification_result["details"].append("Face verification unavailable")
        elif not selfie_path:
            face_verification_result["details"].append("Selfie not provided")
        else:
            id_image_path = uploaded_paths[0]
            is_verified, match_score, cropped_faces = face_verifier.verify_faces(
                id_image_path, selfie_path, output_base_dir
            )
            cropped_face_paths.extend(cropped_faces)
            
            is_alive = False
            if liveness_checker:
                is_alive = liveness_checker.check_liveness(selfie_path)
                face_verification_result["details"].append(
                    f"Liveness check {'passed' if is_alive else 'failed'}"
                )
            else:
                face_verification_result["details"].append("Liveness check unavailable")
            
            final_verified = is_verified and is_alive
            face_verification_result["is_verified"] = final_verified
            face_verification_result["match_score"] = round(match_score, 2)
            face_verification_result["is_alive"] = is_alive
            face_verification_result["details"].append(
                f"Face verification {'successful' if final_verified else 'failed'}"
            )
        
        final_result = {
            "document_type": combined_result["document_type"],
            "fields": combined_result["fields"],
            "is_face_verified": face_verification_result["is_verified"],
            "face_match_score": face_verification_result["match_score"]
        }
        
        return {
            "status": "success",
            "combined_result": final_result,
            "processed_files": uploaded_paths,
            "annotated_files": annotated_paths,
            "preprocessed_files": preprocessed_paths,
            "cropped_face_files": cropped_face_paths,
            "face_verification": face_verification_result
        }
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "combined_result": {},
            "processed_files": [],
            "annotated_files": [],
            "preprocessed_files": [],
            "cropped_face_files": [],
            "face_verification": {
                "is_verified": False,
                "match_score": 0.0,
                "is_alive": False,
                "details": [str(e)]
            }
        }

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint directing to Gradio UI.
    """
    return """
    <html>
        <head>
            <title>Smart ID Extraction API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }
                h1 { color: #333; }
                p { color: #666; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Smart ID Extraction API</h1>
            <p>Welcome to the Smart ID Extraction API.</p>
            <p>Use the Gradio UI at <a href="http://127.0.0.1:7860">http://127.0.0.1:7860</a>.</p>
            <p>API docs: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """

@app.post("/process")
async def process_endpoint(
    doc_type: str = Form(...),
    files: List[UploadFile] = File(...),
    selfie_file: Optional[UploadFile] = File(None)
):
    """
    Endpoint to process images for OCR, face verification, and liveness check.
    
    Args:
        doc_type: Document type ("Citizenship", "License", "Passport").
        files: List of ID image files.
        selfie_file: Optional selfie image.
    
    Returns:
        JSON response with results.
    """
    allowed_types = ["image/jpeg", "image/png"]
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only JPEG/PNG allowed."
            )
    if selfie_file and selfie_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {selfie_file.filename}. Only JPEG/PNG allowed."
        )
    
    result = await process_document(files, doc_type, self
```python
# Continuation of app.py (previous message was cut off)

    return JSONResponse(content=result)

@app.get("/status")
async def status():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# ---- gradio_ui.py ----
import gradio as gr
import requests
import json
import os
import shutil
from typing import List, Dict, Any

def process_images(doc_type: str, id_images: List[str], selfie_image: str) -> Dict[str, Any]:
    """
    Call the FastAPI /process endpoint to process images, perform face verification, and liveness check.
    
    Args:
        doc_type: Type of document ("Citizenship", "License", "Passport").
        id_images: List of ID image file paths.
        selfie_image: Path to the selfie image (or None if not provided).
    
    Returns:
        Dictionary with JSON output, verification results, and gallery images.
    """
    url = "http://127.0.0.1:8000/process"
    
    files = [
        ('files', (os.path.basename(img_path), open(img_path, 'rb'), 'image/jpeg'))
        for img_path in id_images
    ]
    if selfie_image:
        files.append(
            ('selfie_file', (os.path.basename(selfie_image), open(selfie_image, 'rb'), 'image/jpeg'))
        )
    
    data = {"doc_type": doc_type}
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        
        json_output = json.dumps(result["combined_result"], indent=2)
        face_verification = result["face_verification"]
        verification_output = (
            f"Face Verification: {'Verified' if face_verification['is_verified'] else 'Not Verified'}\n"
            f"Match Score: {face_verification['match_score']:.2f}\n"
            f"Liveness Check: {'Passed' if face_verification['is_alive'] else 'Failed'}\n"
            f"Details:\n" + "\n".join(face_verification["details"])
        )
        
        gallery_images = []
        if result["status"] == "success":
            for img_path in result["preprocessed_files"]:
                gallery_images.append((img_path, "Preprocessed"))
            for img_path in result["annotated_files"]:
                gallery_images.append((img_path, "Annotated"))
            for img_path in result["cropped_face_files"]:
                gallery_images.append((img_path, "Cropped Face"))
        
        return {
            "json_output": json_output,
            "verification_output": verification_output,
            "gallery": gallery_images if gallery_images else [(None, "No images generated")],
            "error": None
        }
    except requests.exceptions.RequestException as e:
        return {
            "json_output": "",
            "verification_output": "",
            "gallery": [],
            "error": f"Error calling API: {str(e)}"
        }
    finally:
        for _, file_tuple in files:
            file_tuple[1].close()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
        # Smart ID Extraction System
        Upload ID documents and an optional selfie to extract information, verify identity, and check liveness.
        View preprocessed, annotated, and cropped face images in the gallery below.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            doc_type = gr.Dropdown(
                choices=["Citizenship", "License", "Passport"],
                label="Document Type",
                value="Citizenship"
            )
            id_image_input = gr.File(
                label="Upload ID Images (JPEG/PNG)",
                file_types=[".jpg", ".jpeg", ".png"],
                file_count="multiple"
            )
            selfie_input = gr.File(
                label="Upload Selfie (Optional, JPEG/PNG)",
                file_types=[".jpg", ".jpeg", ".png"],
                file_count="single"
            )
            process_button = gr.Button("Process", variant="primary")
        
        with gr.Column(scale=2):
            json_output = gr.Textbox(label="Extracted Information (JSON)", lines=15, interactive=False)
            verification_output = gr.Textbox(label="Verification Results", lines=5, interactive=False)
            error_message = gr.Textbox(label="Error", lines=2, interactive=False, visible=False)
    
    gallery = gr.Gallery(
        label="Processed Images",
        columns=3,
        height="auto",
        object_fit="contain",
        show_label=True,
        preview=True
    )
    
    def handle_process(doc_type: str, id_images: List, selfie_image: str):
        """
        Handle the process button click, processing images and updating outputs.
        """
        if not id_images:
            return "", "", [], "Please upload at least one ID image."
        
        id_image_paths = [img.name for img in id_images] if isinstance(id_images, list) else []
        selfie_path = selfie_image.name if selfie_image else None
        
        result = process_images(doc_type, id_image_paths, selfie_path)
        return (
            result["json_output"],
            result["verification_output"],
            result["gallery"],
            result["error"] or ""
        )
    
    process_button.click(
        fn=handle_process,
        inputs=[doc_type, id_image_input, selfie_input],
        outputs=[json_output, verification_output, gallery, error_message]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

# ---- face_verifier.py ----
import numpy as np
from insightface.app import FaceAnalysis
import cv2
import os
import logging
from typing import Tuple, Optional, List

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_id_extraction.log")]
)
logger = logging.getLogger(__name__)

class FaceVerifier:
    def __init__(self):
        """
        Initialize InsightFace model for face detection and embedding.
        """
        try:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.debug("InsightFace initialized")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            logger.debug("Haar Cascade loaded")
        except Exception as e:
            logger.error(f"FaceVerifier initialization failed: {str(e)}")
            raise
    
    def preprocess_for_detection(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve face detection.
        
        Args:
            img: Input image as numpy array.
        
        Returns:
            Preprocessed image.
        """
        try:
            max_size = 1280
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return img
    
    def detect_and_embed(self, image_path: str, output_dir: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Detect face, generate embedding, and save cropped face.
        
        Args:
            image_path: Path to image file.
            output_dir: Directory to save cropped face.
        
        Returns:
            Tuple of (face embedding, cropped face path) or (None, None).
        """
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
                    logger.warning(f"No face detected in {image_path}")
                    return None, None
                
                (x, y, w, h) = faces[0]
                bbox = [x, y, x + w, y + h]
                cropped_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                faces = self.app.get(cropped_face_rgb)
                if not faces:
                    logger.warning(f"Embedding failed for cropped face in {image_path}")
                    return None, None
                embedding = faces[0].embedding
                logger.debug(f"Haar Cascade detected face, embedding generated")
            
            face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            face_path = os.path.join(output_dir, f"cropped_face_{os.path.basename(image_path)}")
            cv2.imwrite(face_path, face_img)
            logger.debug(f"Cropped face saved to {face_path}")
            return embedding, face_path
        
        except Exception as e:
            logger.error(f"Face detection error in {image_path}: {str(e)}")
            return None, None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding.
            embedding2: Second face embedding.
        
        Returns:
            Similarity score (0.0–1.0).
        """
        try:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            similarity = float(np.clip(np.dot(embedding1, embedding2), 0.0, 1.0))
            logger.debug(f"Similarity score: {similarity}")
            return similarity
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            return 0.0
    
    def verify_faces(self, id_image_path: str, selfie_image_path: str, output_dir: str, threshold: float = 0.6) -> Tuple[bool, float, List[str]]:
        """
        Verify if ID face matches selfie face.
        
        Args:
            id_image_path: Path to ID image.
            selfie_image_path: Path to selfie image.
            output_dir: Directory to save cropped faces.
            threshold: Similarity threshold.
        
        Returns:
            Tuple of (is_verified, match_score, cropped_face_paths).
        """
        cropped_face_paths = []
        try:
            id_embedding, id_face_path = self.detect_and_embed(id_image_path, output_dir)
            if id_embedding is None:
                logger.warning("No face detected in ID image")
                return False, 0.0, []
            if id_face_path:
                cropped_face_paths.append(id_face_path)
            
            selfie_embedding, selfie_face_path = self.detect_and_embed(selfie_image_path, output_dir)
            if selfie_embedding is None:
                logger.warning("No face detected in selfie image")
                return False, 0.0, cropped_face_paths
            if selfie_face_path:
                cropped_face_paths.append(selfie_face_path)
            
            match_score = self.compute_similarity(id_embedding, selfie_embedding)
            is_verified = match_score >= threshold
            logger.debug(f"Verification: is_verified={is_verified}, score={match_score}")
            return is_verified, match_score, cropped_face_paths
        
        except Exception as e:
            logger.error(f"Face verification failed: {str(e)}")
            return False, 0.0, cropped_face_paths

# ---- liveness_checker.py ----
import cv2
import dlib
import numpy as np
import logging
from typing import bool

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_id_extraction.log")]
)
logger = logging.getLogger(__name__)