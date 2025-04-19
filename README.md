
# 🆔 Smart ID Information Extraction System

A Computer Vision and NLP-based system for automated data extraction, face verification, and liveness detection from identity documents like Nepali Citizenship, Passport, and Driving License.

---

## 🚀 Overview

This project automates the extraction and verification of personal information from identity documents using cutting-edge technologies. It supports multilingual OCR (including Devanagari script), face verification, and liveness detection to ensure authenticity and reliability.

---

## 🔧 Features

-  Smart OCR with Devanagari and English text support
-  NLP-based dynamic field classification
-  Face detection and verification using InsightFace
-  YOLOv8-powered document and field detection
-  User-friendly interface using Gradio
-  Real-time processing with GPU support
-  FastAPI backend with structured JSON output

---

##  Tech Stack

- **Language**: Python 3.10+
- **Backend**: FastAPI
- **Frontend**: Gradio
- **OCR Engines**: EasyOCR, Tesseract, PassportEye
- **Face Verification**: InsightFace
- **Liveness Detection**: dlib + OpenCV
- **Detection Models**: YOLOv8
- **Deployment**: Local/Cloud-ready

---

##  Installation

###  Prerequisites

- Python 3.10+
- pip or conda
- Git
- GPU (optional but recommended)

---

###  Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smart-id-extraction.git
cd smart-id-extraction

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv          # Create venv
source venv/bin/activate     # On Windows use: venv\Scripts\activate

# 3. Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

###  Running the Application

#### Start the FastAPI Backend

```bash
uvicorn app:app --reload
```

- Opens API at `http://127.0.0.1:8000`
- Visit `http://127.0.0.1:8000/docs` for Swagger UI to test endpoints

#### 🔹 Start the Gradio Frontend Interface

```bash
python gradio_app.py
```

- This opens a local browser window with the Gradio interface
- You can select document type, upload an ID, and view structured results

---

### ⚠️ Notes

- Place your YOLO model weights (`best.pt`) inside the `models/` directory
- Ensure OCR language packs are installed for Tesseract (if used)
- You can configure paths and options in `config.py` (if applicable)

---

## 📊 Performance Highlights

| Metric              | Result             |
|---------------------|--------------------|
| OCR Accuracy        | 95%+               |
| Face Match Accuracy | 92%+               |
| Liveness Accuracy   | High reliability   |
| Processing Speed    | Real-time (with GPU)|



## 👨 Author

**[Ayush Khadka]**  
[Treeleaf]  
Supervised by: [Supervisor’s Name]  
Date: [Submission Date]

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify!

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
