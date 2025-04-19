
# 🆔 Smart ID Information Extraction System

A Computer Vision and NLP-based system for automated data extraction, face verification, and liveness detection from identity documents such as Nepali Citizenship, Passport, and Driving License.

---

##  Overview

This project automates the extraction and verification of personal information from identity documents using a pipeline of advanced image preprocessing, deep learning models, OCR engines, and NLP techniques. It supports multilingual text (Devanagari + English), verifies identity using face matching, and detects liveness using blink-based detection — all wrapped in a user-friendly interface.

---

##  Features

-  OCR FOR NEPALI DOCUMENT EXTRACTION
-  NLP-based smart field classification
-  YOLOv8 for document & field detection
-  Face verification using InsightFace
-  Robust image preprocessing (rotation, enhancement, cropping)
-  Gradio-based interactive frontend
-  FastAPI backend with real-time JSON output
-  Optimized for GPU acceleration

---

##  Tech Stack

| Component     | Technology           |
|---------------|----------------------|
| Language      | Python 3.10+         |
| Backend       | FastAPI              |
| Frontend      | Gradio               |
| OCR Engines   | EasyOCR, Tesseract, PassportEye |
| Face Match    | InsightFace          |
| Liveness Check| dlib + OpenCV        |
| Detection     | YOLOv8               |
| Dev Tools     | VS Code, Git, pip    |

---


###  Prerequisites

- Python 3.10+
- pip or conda
- Git
- Optional: GPU + CUDA for real-time performance

---

### 🧪 Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smart-id-extraction.git
cd smart-id-extraction

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
# OR
source venv/bin/activate      # On macOS/Linux

# 3. Move into the working directory
cd smartpreprocessing

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

###  Running the App

####  Start FastAPI Backend

```bash
python app.py
```

- Runs at `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

#### 🔹 Start Gradio Frontend

```bash
python gradio_ui.py
```

- Opens browser UI for document upload, field extraction, and face match results


---

## 📊 Performance Highlights

| Metric              | Result             |
|---------------------|--------------------|
| OCR Accuracy        | 72%+               |
| Face Match Accuracy | 92%+               |
| Liveness Accuracy   | High reliability   |
| Processing Speed    | Real-time (GPU)    |

---


### OCR Accuracy
```bash
cd smartpreprocessing/evaluation
python evaluate_ocr.py
```




## 👨‍💻 Author & Mentorship

**Author:** [Your Full Name]  
**Institution:** [Your Institution or Organization]  
**Supervisor:** [Supervisor Name]  
**Date:** [Submission Date]

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with proper credit.

---

## 🔗 References

- [FastAPI](https://fastapi.tiangolo.com)
- [Gradio](https://www.gradio.app)
- [InsightFace](https://github.com/deepinsight/insightface)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [dlib](http://dlib.net)
- [YOLOv8](https://github.com/ultralytics/ultralytics)

---
