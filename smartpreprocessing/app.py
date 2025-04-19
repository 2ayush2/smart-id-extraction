#finallll apiiiii
import os
import json
import tempfile
import shutil
import logging
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from utils.preprocess import preprocess_image, preprocess_selfie
from utils.passport_ocr import process_image as process_passport
from utils.citizenship_ocr import process_image as process_citizenship
from utils.license_ocr import process_image as process_license
from utils.selfie_ocr import process_image as process_selfie
from utils.combine_results import combine_results
from utils.llm_ocr import process_image as process_llm_ocr

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("smart_id_extraction.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Initializing FastAPI server")

app = FastAPI(title="Smart ID Extraction API")

# Mount static files
processed_dir = "processed"
os.makedirs(processed_dir, exist_ok=True)
app.mount("/processed", StaticFiles(directory=processed_dir), name="processed")
logger.info("Mounted static files at /processed")

# Custom JSON encoder to handle bool and other types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, bool):
            return str(obj).lower()
        return super().default(obj)

# Custom exception handler
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        media_type="application/json"
    )

def verify_faces(selfie: np.ndarray, id_image: np.ndarray) -> Dict[str, Any]:
    try:
        from utils.preprocess import FACE_APP
        if FACE_APP is None:
            logger.warning("FACE_APP not available")
            return {
                "is_verified": "false",
                "match_score": 0.0,
                "details": ["Face verification not available"]
            }
        selfie_rgb = cv2.cvtColor(selfie, cv2.COLOR_BGR2RGB)
        id_rgb = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
        selfie_faces = FACE_APP.get(selfie_rgb)
        id_faces = FACE_APP.get(id_rgb)
        logger.debug(f"Selfie faces: {len(selfie_faces)}, ID faces: {len(id_faces)}")
        if not selfie_faces or not id_faces:
            logger.warning("No faces detected in selfie or ID")
            return {
                "is_verified": "false",
                "match_score": 0.0,
                "details": ["No faces detected"]
            }
        selfie_embedding = selfie_faces[0].normed_embedding
        id_embedding = id_faces[0].normed_embedding
        similarity = np.dot(selfie_embedding, id_embedding)
        is_verified = similarity > 0.4
        return {
            "is_verified": str(is_verified).lower(),
            "match_score": round(float(similarity), 2),
            "details": [f"Similarity score: {similarity:.2f}"]
        }
    except Exception as e:
        logger.error(f"Face verification failed: {str(e)}")
        return {
            "is_verified": "false",
            "match_score": 0.0,
            "details": [f"Face verification failed: {str(e)}"]
        }

def save_to_final_combined(run_id: str, doc_type: str, combined_result: Dict[str, Any]):
    final_json_path = os.path.join(processed_dir, run_id, doc_type.lower(), "final_combined.json").replace("\\", "/")
    combined_data = []
    if os.path.exists(final_json_path):
        try:
            with open(final_json_path, "r", encoding="utf-8") as f:
                combined_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read final_combined.json: {str(e)}")
    
    for entry in combined_data:
        if entry["run_id"] == run_id:
            logger.info(f"Skipping duplicate run_id: {run_id}")
            return
    
    fields = {}
    confidences = combined_result.get("confidences", {})
    for field, value in combined_result.get("fields", {}).items():
        if field != "error":
            fields[field] = value["ocr_text"] if isinstance(value, dict) else value
        else:
            fields[field] = value
    
    new_entry = {
        "run_id": run_id,
        "document_type": doc_type,
        "timestamp": datetime.now().isoformat(),
        "fields": fields,
        "confidences": confidences
    }
    combined_data.append(new_entry)
    
    try:
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False, cls=CustomJSONEncoder)
        logger.info(f"Saved results to final_combined.json for run_id: {run_id}")
    except Exception as e:
        logger.error(f"Failed to save final_combined.json: {str(e)}")

async def process_document(
    files: List[UploadFile],
    doc_type: str,
    selfie_file: Optional[UploadFile] = None
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(processed_dir, run_id, doc_type.lower()).replace("\\", "/")
    preprocess_dir = os.path.join(output_dir, "preprocess").replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)
    results = []
    preprocessed_files = []
    annotated_files = []
    cropped_face_files = []
    face_verification = {
        "is_verified": "false",
        "match_score": 0.0,
        "details": ["No selfie provided"]
    }
    
    try:
        if not files:
            logger.error("No ID images provided")
            raise HTTPException(status_code=400, detail="No ID images provided")
        if doc_type not in ["Passport", "Citizenship", "License", "Generic", "Llm ocr"]:
            logger.error(f"Invalid document type: {doc_type}")
            raise HTTPException(status_code=400, detail=f"Invalid document type: {doc_type}")
        
        logger.info(f"Processing {len(files)} images for {doc_type} (Run ID: {run_id})")
        
        # Process ID images
        best_img_path = None
        for file in files:
            file_path = os.path.join(output_dir, file.filename).replace("\\", "/")
            try:
                with open(file_path, "wb") as f:
                    content = await file.read()
                    if not content:
                        logger.warning(f"ID image {file.filename} is empty, skipping")
                        continue
                    f.write(content)
                    logger.debug(f"ID image saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save ID image {file.filename}: {str(e)}")
                continue
            
            # Preprocess image
            binarized_path, orig_best_img_path, face_path, preproc_images = preprocess_image(
                file_path, preprocess_dir, output_suffix=file.filename
            )
            if not preproc_images:
                logger.warning(f"Preprocessing failed for {file_path}, skipping")
                continue
            
            # Process original image
            img_path = orig_best_img_path or file_path
            try:
                if doc_type == "Passport":
                    result = process_passport(img_path, output_dir)
                elif doc_type == "Citizenship":
                    result = process_citizenship(img_path, output_dir)
                elif doc_type == "License":
                    result = process_license(img_path, output_dir)
                elif doc_type == "Llm ocr":
                    result = process_llm_ocr(img_path, output_dir)
                else:
                    logger.warning(f"Unsupported doc_type: {doc_type}, using fallback OCR")
                    result = process_citizenship(img_path, output_dir)

                if result and "error" not in result.get("fields", {}):
                    score = result.get("score", 0)
                    logger.debug(f"OCR score for {img_path}: {score}")
                    # Temporarily bypass score threshold for debugging
                    logger.info(f"Accepting {file.filename} for face verification (bypassing score)")
                    best_img_path = img_path
                    results.append(result)
                    result_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}.json").replace("\\", "/")
                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=4, cls=CustomJSONEncoder)
                    preprocessed_files.append(os.path.join(run_id, doc_type.lower(), "preprocess", os.path.basename(img_path)).replace("\\", "/"))
                    if face_path:
                        cropped_face_files.append(os.path.join(run_id, doc_type.lower(), "preprocess", os.path.basename(face_path)).replace("\\", "/"))
                    annotated_filename = f"annotated_{os.path.basename(img_path)}"
                    annotated_path = os.path.join(output_dir, annotated_filename).replace("\\", "/")
                    if os.path.exists(annotated_path):
                        annotated_files.append(os.path.join(run_id, doc_type.lower(), annotated_filename).replace("\\", "/"))
                else:
                    logger.warning(f"No valid OCR result for {file.filename}")
            except Exception as e:
                logger.error(f"OCR processing failed for {img_path}: {str(e)}")
                continue
            
            # If no best_img_path, try rotated images
            if not best_img_path and preproc_images:
                for img in preproc_images:
                    if "rotated_90" in img["path"]:
                        img_path = img["path"]
                        try:
                            if doc_type == "Passport":
                                result = process_passport(img_path, output_dir)
                            elif doc_type == "Citizenship":
                                result = process_citizenship(img_path, output_dir)
                            elif doc_type == "License":
                                result = process_license(img_path, output_dir)
                            elif doc_type == "Llm ocr":
                                result = process_llm_ocr(img_path, output_dir)
                            else:
                                result = process_citizenship(img_path, output_dir)
                            if result and "error" not in result.get("fields", {}):
                                logger.info(f"Accepting rotated {file.filename} for face verification")
                                best_img_path = img_path
                                results.append(result)
                                result_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}.json").replace("\\", "/")
                                with open(result_path, "w", encoding="utf-8") as f:
                                    json.dump(result, f, indent=4, cls=CustomJSONEncoder)
                                preprocessed_files.append(os.path.join(run_id, doc_type.lower(), "preprocess", os.path.basename(img_path)).replace("\\", "/"))
                                if face_path:
                                    cropped_face_files.append(os.path.join(run_id, doc_type.lower(), "preprocess", os.path.basename(face_path)).replace("\\", "/"))
                                annotated_filename = f"annotated_{os.path.basename(img_path)}"
                                annotated_path = os.path.join(output_dir, annotated_filename).replace("\\", "/")
                                if os.path.exists(annotated_path):
                                    annotated_files.append(os.path.join(run_id, doc_type.lower(), annotated_filename).replace("\\", "/"))
                            else:
                                logger.warning(f"No valid OCR result for rotated {file.filename}")
                        except Exception as e:
                            logger.error(f"OCR processing failed for {img_path}: {str(e)}")
                            continue
                        break
        
        # Process selfie
        selfie_best_img = None
        if selfie_file:
            selfie_path = os.path.join(output_dir, "selfie.jpg").replace("\\", "/")
            try:
                with open(selfie_path, "wb") as f:
                    content = await selfie_file.read()
                    if not content:
                        logger.warning("Selfie file is empty")
                        face_verification["details"] = ["Selfie file is empty"]
                    else:
                        f.write(content)
                        logger.debug(f"Selfie saved to {selfie_path}")
                        if not os.path.exists(selfie_path):
                            logger.error(f"Selfie file not saved: {selfie_path}")
                            face_verification["details"] = ["Selfie file not saved"]
                        else:
                            _, selfie_best_path, selfie_face_path, selfie_preproc = preprocess_selfie(
                                selfie_path, preprocess_dir, output_suffix="selfie"
                            )
                            logger.debug(f"Selfie best path: {selfie_best_path}, face path: {selfie_face_path}")
                            if selfie_best_path:
                                if not os.path.exists(selfie_best_path):
                                    logger.error(f"Selfie best path does not exist: {selfie_best_path}")
                                    face_verification["details"] = ["Selfie preprocessing failed: file not found"]
                                else:
                                    selfie_best_img = cv2.imread(selfie_best_path)
                                    logger.debug(f"Selfie best img loaded: {selfie_best_img is not None}, shape: {selfie_best_img.shape if selfie_best_img is not None else 'None'}")
                                    if selfie_best_img is None:
                                        face_verification["details"] = ["Failed to load preprocessed selfie"]
                            if selfie_face_path:
                                cropped_face_files.append(os.path.join(run_id, doc_type.lower(), "preprocess", os.path.basename(selfie_face_path)).replace("\\", "/"))
                            for img in selfie_preproc or []:
                                preprocessed_files.append(os.path.join(run_id, doc_type.lower(), "preprocess", os.path.basename(img["path"])).replace("\\", "/"))
                            selfie_result = process_selfie(selfie_path, output_dir)
                            logger.debug(f"Selfie OCR result: {selfie_result}")
                            if selfie_result and "error" not in selfie_result.get("fields", {}):
                                results.append(selfie_result)
                            else:
                                logger.warning("Selfie OCR failed or no face detected")
            except Exception as e:
                logger.error(f"Selfie processing failed: {str(e)}")
                face_verification["details"] = [f"Selfie processing failed: {str(e)}"]
        
        # Face verification
        if selfie_best_img is not None and best_img_path:
            id_image = cv2.imread(best_img_path)
            logger.debug(f"ID image loaded: {id_image is not None}, shape: {id_image.shape if id_image is not None else 'None'}")
            if id_image is not None:
                logger.debug("Calling verify_faces")
                face_verification = verify_faces(selfie_best_img, id_image)
                logger.debug(f"Face verification result: {face_verification}")
            else:
                face_verification["details"] = ["Invalid ID image for verification"]
        else:
            logger.debug(f"Skipping face verification: selfie_best_img={selfie_best_img is not None}, best_img_path={best_img_path}")
        
        # Combine results
        try:
            combined_result = combine_results(doc_type, output_dir)
        except Exception as e:
            logger.error(f"Combine results failed: {str(e)}")
            combined_result = {"fields": {"error": str(e)}, "confidences": {}}
        
        if "error" not in combined_result.get("fields", {}):
            save_to_final_combined(run_id, doc_type, combined_result)
        
        # Prepare response
        response_fields = {}
        response_confidences = {}
        for field, value in combined_result.get("fields", {}).items():
            if field != "error":
                response_fields[field] = value["ocr_text"] if isinstance(value, dict) else value
                response_confidences[field] = combined_result["confidences"].get(field, 0.0)
            else:
                response_fields[field] = value
        
        response = {
            "combined_result": {
                "fields": response_fields,
                "confidences": response_confidences
            },
            "face_verification": face_verification,
            "preprocessed_files": [f"/processed/{path}" for path in preprocessed_files],
            "annotated_files": [f"/processed/{path}" for path in annotated_files],
            "cropped_face_files": [f"/processed/{path}" for path in cropped_face_files]
        }
        logger.debug(f"Response prepared: {response}")
        return response
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Processing failed for Run ID {run_id}: {str(e)}", exc_info=True)
        return {
            "combined_result": {"fields": {"error": str(e)}, "confidences": {}},
            "face_verification": {
                "is_verified": "false",
                "match_score": 0.0,
                "details": [f"Processing failed: {str(e)}"]
            },
            "preprocessed_files": [],
            "annotated_files": [],
            "cropped_face_files": []
        }
    
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")

@app.post("/process")
async def process_endpoint(
    doc_type: str = Form(...),
    files: List[UploadFile] = File(...),
    selfie_file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    logger.info("Received POST /process request")
    allowed_types = ["image/jpeg", "image/png"]
    max_file_size = 5 * 1024 * 1024
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only JPEG/PNG allowed.")
        if file.size > max_file_size:
            raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 5MB limit.")
    if selfie_file and selfie_file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid selfie file type: {selfie_file.filename}.")
    if selfie_file and selfie_file.size > max_file_size:
        raise HTTPException(status_code=400, detail=f"Selfie file exceeds 5MB limit.")
    
    result_dict = await process_document(files, doc_type, selfie_file)
    logger.debug(f"POST /process response: {result_dict}")
    return JSONResponse(content=result_dict)

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to serve index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load interface")

@app.get("/status")
async def status():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}