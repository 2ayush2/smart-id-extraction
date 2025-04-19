import cv2
from PIL import Image
import numpy as np

def correct_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    angle = 0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(image_path: str) -> Image.Image:
    image = cv2.imread(image_path)
    rotated = correct_rotation(image)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = thresh.shape
    scale = 100
    if h < 500 or w < 500:
        scale = 200
    elif h > 2000 or w > 2000:
        scale = 50
    resized = cv2.resize(thresh, (int(w * scale / 100), int(h * scale / 100)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)
