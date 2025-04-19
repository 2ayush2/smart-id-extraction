import os
import json
import logging
from typing import Dict, Any
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("combine_results.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def combine_results(doc_type: str, output_dir: str) -> Dict[str, Any]:
    """Combine OCR results from all processed images."""
    try:
        # Validate document type
        valid_doc_types = ["Citizenship", "License", "Passport", "Generic", "Llm ocr"]
        if doc_type not in valid_doc_types:
            logger.error(f"Invalid document type: {doc_type}")
            return {"fields": {"error": {"ocr_text": f"Invalid document type: {doc_type}", "confidence": 0.0}}, "confidences": {"error": 0.0}}

        fields_dir = output_dir
        combined_fields = {}
        confidences = {}
        seen_values = {}

        # Look for result_*.json files (matching process_document)
        json_files = [f for f in os.listdir(fields_dir) if f.startswith("result_") and f.endswith(".json")]
        if not json_files:
            logger.warning(f"No result_*.json files found in {fields_dir}")
            return {"fields": {"error": {"ocr_text": "No OCR results found", "confidence": 0.0}}, "confidences": {"error": 0.0}}

        logger.info(f"Processing {len(json_files)} JSON files for {doc_type}")

        for json_file in json_files:
            json_path = os.path.join(fields_dir, json_file).replace("\\", "/")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    result = json.load(f)

                # Validate JSON structure
                if not isinstance(result, dict):
                    logger.warning(f"Skipping malformed JSON in {json_file}: expected dict, got {type(result)}")
                    continue

                fields = result.get("fields", {})
                result_confidences = result.get("confidences", {})

                if not isinstance(fields, dict) or not isinstance(result_confidences, dict):
                    logger.warning(f"Skipping invalid fields/confidences in {json_file}")
                    continue

                logger.debug(f"Fields in {json_file}: {fields.keys()}")

                # Process each field
                for field, value in fields.items():
                    if field == "error" or not isinstance(value, dict):
                        logger.debug(f"Skipping field {field} in {json_file}: invalid format")
                        continue
                    ocr_text = value.get("ocr_text", "")
                    confidence = float(result_confidences.get(field, 0.0))
                    source_image = value.get("source_image", json_file)
                    if not ocr_text:
                        logger.debug(f"Skipping field {field} in {json_file}: empty ocr_text")
                        continue

                    # Initialize field if not seen
                    if field not in seen_values:
                        seen_values[field] = []

                    # Adjust fuzzy matching threshold based on field length
                    fuzzy_threshold = 85 if len(ocr_text) > 5 else 95  # Stricter for short fields

                    # Skip fuzzy matching for strict passport fields
                    skip_fuzzy = doc_type == "Passport" and field.lower() in [
                        "passport_number", "sex", "passport_type", "personal_number"
                    ]

                    if skip_fuzzy:
                        seen_values[field].append({
                            "ocr_text": ocr_text,
                            "confidence": confidence,
                            "crop_image_path": value.get("crop_image_path", ""),
                            "source_image": source_image
                        })
                    else:
                        # Check for similar values using fuzzy matching
                        matched = False
                        for stored in seen_values[field]:
                            if fuzz.ratio(ocr_text, stored["ocr_text"]) > fuzzy_threshold:
                                if confidence > stored["confidence"]:
                                    stored["ocr_text"] = ocr_text
                                    stored["confidence"] = confidence
                                    stored["crop_image_path"] = value.get("crop_image_path", "")
                                    stored["source_image"] = source_image
                                matched = True
                                break
                        if not matched:
                            seen_values[field].append({
                                "ocr_text": ocr_text,
                                "confidence": confidence,
                                "crop_image_path": value.get("crop_image_path", ""),
                                "source_image": source_image
                            })

            except Exception as e:
                logger.error(f"Failed to process {json_file}: {str(e)}")
                continue

        # Select best value for each field
        for field, values in seen_values.items():
            if values:
                best = max(values, key=lambda x: x["confidence"])
                combined_fields[field] = {
                    "ocr_text": best["ocr_text"],
                    "confidence": best["confidence"],
                    "crop_image_path": best["crop_image_path"],
                    "source_image": best["source_image"]
                }
                confidences[field] = best["confidence"]
            else:
                logger.debug(f"No values for field {field}")

        if not combined_fields:
            logger.warning("No valid fields combined")
            return {"fields": {"error": {"ocr_text": "No valid OCR fields found", "confidence": 0.0}}, "confidences": {"error": 0.0}}

        logger.info(f"Combined {len([f for f in combined_fields if combined_fields[f]['ocr_text']])} fields for {doc_type}: {list(combined_fields.keys())}")
        return {"fields": combined_fields, "confidences": confidences}

    except Exception as e:
        logger.error(f"Error combining results: {str(e)}")
        return {"fields": {"error": {"ocr_text": str(e), "confidence": 0.0}}, "confidences": {"error": 0.0}}