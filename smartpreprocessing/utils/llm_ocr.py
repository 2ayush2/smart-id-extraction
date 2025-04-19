import os
import cv2
import json
import requests
from groq import Groq
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

def upload_to_uguu(image_path: str) -> str:
    """Upload image to uguu.se and return the URL."""
    try:
        with open(image_path, "rb") as f:
            response = requests.post("https://uguu.se/upload.php", files={"files[]": f})
            if response.status_code == 200:
                return response.json()["files"][0]["url"]
            else:
                logger.error(f"Upload failed: {response.text}")
                raise Exception("Image upload to uguu.se failed")
    except Exception as e:
        logger.error(f"Upload to uguu.se failed: {str(e)}")
        raise

def process_image(img_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Process an image using Groq's LLM to extract ID information.
    Args:
        img_path: Path to the input image.
        output_dir: Directory to save outputs (e.g., annotated image, JSON).
    Returns:
        Dictionary with fields, confidences, and annotated path.
    """
    try:
        # Load and verify image
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            return {
                "fields": {"error": {"ocr_text": "Failed to load image", "confidence": 0.0}},
                "confidences": {"error": 0.0},
                "annotated_path": ""
            }

        file_name = os.path.basename(img_path)
        annotated_path = os.path.join(output_dir, f"annotated_{file_name}")

        # Upload image to uguu.se
        try:
            image_url = upload_to_uguu(img_path)
            logger.debug(f"Uploaded image to: {image_url}")
        except Exception as e:
            logger.error(f"Failed to upload image: {str(e)}")
            return {
                "fields": {"error": {"ocr_text": "Image upload failed", "confidence": 0.0}},
                "confidences": {"error": 0.0},
                "annotated_path": ""
            }

        # Call Groq API
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": """Extract the following information from the image in the same language as the original.
                    Return a JSON object with the following fields:
                    {
                        "full_name": "",
                        "date_of_birth": "",
                        "national_id": "",
                        "gender": "",
                        "expiry_date": "",
                        "address": "",
                        "nationality": ""
                    }
                    If a field is not found, return an empty string."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        # Parse Groq response, removing Markdown formatting
        result_text = completion.choices[0].message.content
        
        if result_text.startswith("```json"):
            result_text = result_text[len("```json"):].strip()
        if result_text.endswith("```"):
            result_text = result_text[:-len("```")].strip()

        try:
            extracted_data = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response after cleaning: {result_text}, error: {str(e)}")
            return {
                "fields": {"error": {"ocr_text": "Invalid Groq response", "confidence": 0.0}},
                "confidences": {"error": 0.0},
                "annotated_path": ""
            }

     
        field_mapping = {
            "full_name": "Name",
            "date_of_birth": "Date of Birth",
            "national_id": "National ID",
            "gender": "Gender",
            "expiry_date": "Expiry Date",
            "address": "Address",
            "nationality": "Nationality"
        }

        fields = {}
        confidences = {}
        for groq_field, pipeline_field in field_mapping.items():
            ocr_text = extracted_data.get(groq_field, "")
            fields[pipeline_field] = {
                "ocr_text": ocr_text,
                "confidence": 0.9 if ocr_text else 0.0,  
                "crop_image_path": "",
                "source_image": file_name
            }
            confidences[pipeline_field] = 0.9 if ocr_text else 0.0

        
        cv2.imwrite(annotated_path, image)
        logger.debug(f"Saved annotated image: {annotated_path}")

        fields_json_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_fields.json")
        with open(fields_json_path, "w", encoding="utf-8") as f:
            json.dump(fields, f, indent=4, ensure_ascii=False)

        return {
            "fields": fields,
            "confidences": confidences,
            "annotated_path": annotated_path,
            "score": 0  
        }

    except Exception as e:
        logger.error(f"LLM OCR processing failed for {img_path}: {str(e)}")
        return {
            "fields": {"error": {"ocr_text": str(e), "confidence": 0.0}},
            "confidences": {"error": 0.0},
            "annotated_path": ""
        }