import pytesseract
from PIL import Image
from app.utils.image_preprocess import preprocess_image

class DocumentClassifier:
    def classify(self, image_path: str) -> str:
        image = preprocess_image(image_path)
        text = pytesseract.image_to_string(image).lower()
        if "citizenship" in text or "nagrita" in text:
            return "Citizenship"
        elif "passport" in text:
            return "Passport"
        elif "license" in text:
            return "Driving License"
        else:
            return "Unknown"
