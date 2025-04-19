import gradio as gr
import requests
import json
import os
import pandas as pd
from typing import List, Dict, Any, Tuple
from contextlib import ExitStack
import logging

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gradio_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/process")

# Simulated API functions (unchanged from original)
def check_backend_status() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error(f"Backend health check failed: {str(e)}")
        return False

def process_images(doc_type: str, id_image_paths: List[str], selfie_path: str | None, enable_selfie: bool) -> Dict[str, Any]:
    logger.debug(f"Processing document type: {doc_type}, ID images: {id_image_paths}, Selfie: {selfie_path}, Enable selfie: {enable_selfie}")
    
    files: List[Tuple[str, tuple]] = []
    with ExitStack() as stack:
        for path in id_image_paths:
            if path and os.path.exists(path):
                f = stack.enter_context(open(path, 'rb'))
                files.append(('id_images', (os.path.basename(path), f, 'image/jpeg')))
        
        if enable_selfie and selfie_path and os.path.exists(selfie_path):
            f = stack.enter_context(open(selfie_path, 'rb'))
            files.append(('selfie', (os.path.basename(selfie_path), f, 'image/jpeg')))
        
        data = {"doc_type": doc_type.lower(), "enable_selfie": str(enable_selfie).lower()}
        try:
            response = requests.post(API_URL, files=files, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            fields_data = result.get("extracted_fields", [])
            verification_data = result.get("verification", []) if enable_selfie else []
            
            fields_df = pd.DataFrame(fields_data, columns=["Field", "Value", "Confidence"])
            verification_df = pd.DataFrame(verification_data, columns=["Metric", "Value"])
            
            gallery_images = result.get("processed_images", [])
            
            return {
                "fields_df": fields_df,
                "verification_df": verification_df,
                "gallery": gallery_images,
                "error": ""
            }
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "fields_df": pd.DataFrame(),
                "verification_df": pd.DataFrame(),
                "gallery": [],
                "error": f"Processing failed: {str(e)}"
            }

# Gradio UI with fixes
with gr.Blocks(
    theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate", neutral_hue="zinc"),
    css="""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * {
            font-family: 'Inter', 'Roboto', sans-serif !important;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .centered-input {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        }
        .input-card {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 25px;
            background-color: #ffffff;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            width: 100%;
            max-width: 550px;
            transition: transform 0.2s ease;
        }
        .input-card:hover {
            transform: translateY(-2px);
        }
        .image-preview {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 15px;
            padding: 10px;
            background-color: #f9fafb;
            border-radius: 8px;
        }
        .image-preview img {
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
            transition: transform 0.2s ease;
        }
        .image-preview img:hover {
            transform: scale(1.05);
        }
        .selfie-section {
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            transition: background 0.3s ease;
        }
        .selfie-section:hover {
            background: linear-gradient(135deg, #bae6fd 0%, #7dd3fc 100%);
        }
        .image-gallery .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            width: 100%;
            margin: 0 auto;
        }
        .image-gallery .gallery img {
            width: 100%;
            height: 150px;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-gallery .gallery-item label {
            font-size: 14px;
            font-weight: 500;
            color: #333;
            margin-top: 8px;
            text-align: center;
        }
        .error-box {
            color: #d32f2f;
            background-color: #ffebee;
            border-radius: 8px;
            padding: 15px;
            font-weight: 500;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .header-text {
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: 600;
            color: #1a1a1a;
        }
        .compact-input {
            margin-bottom: 15px;
            width: 100%;
        }
        .action-button {
            width: 100%;
            font-size: 16px;
            font-weight: 600;
            padding: 12px;
            border-radius: 8px;
            transition: transform 0.1s ease, box-shadow 0.2s ease;
        }
        .action-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .extract-button {
            background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%) !important;
            color: #ffffff !important;
        }
        .reset-button {
            background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%) !important;
            color: #ffffff !important;
        }
        .selfie-button {
            background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%) !important;
            color: #ffffff !important;
            font-weight: 500 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
        }
        .selfie-button:hover {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        }
        .doc-type-dropdown {
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            padding: 10px !important;
            background-color: #f9fafb !important;
        }
        .file-input {
            border: 2px dashed #d1d5db;
            border-radius: 10px;
            padding: 25px;
            background-color: #f9fafb;
            transition: all 0.2s ease;
            text-align: center;
        }
        .file-input:hover {
            border-color: #60a5fa;
            background-color: #f0f9ff;
        }
        .file-input label {
            color: #4b5563 !important;
            font-weight: 500 !important;
        }
        .output-box {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .output-tabs {
            margin-top: 15px;
        }
        .gr-dataframe table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .gr-dataframe th {
            background-color: #f5f5f5;
            color: #333;
            font-weight: 600;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #e0e0e0;
        }
        .gr-dataframe td {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
            color: #333;
        }
        .gr-dataframe tr:hover {
            background-color: #f9f9f9;
        }
    """
) as demo:
    gr.Markdown(
        """
        # Smart ID Extraction System
        Upload your ID images (Citizenship, License, or Passport) to extract information.  
        Optionally, add a selfie for face verification.  
        """,
        elem_classes=["header-text"]
    )

    # State to control visibility of output sections
    output_visible = gr.State(value=False)

    # Input section (centered initially)
    with gr.Column(elem_classes=["centered-input"], visible=True) as input_column:
        with gr.Group(elem_classes=["input-card"]):
            doc_type = gr.Dropdown(
                choices=["Citizenship", "License", "Passport"],
                label="Document Type",
                value="Citizenship",
                elem_classes=["compact-input", "doc-type-dropdown"]
            )
            id_image_input = gr.File(
                label="Upload ID Images",
                file_types=[".jpg", ".jpeg", ".png"],
                file_count="multiple",
                elem_classes=["compact-input", "file-input"]
            )

            # Small preview of uploaded images (hidden initially)
            image_preview = gr.Gallery(
                label="Uploaded Images Preview",
                height="120px",
                object_fit="cover",
                elem_classes=["image-preview"],
                columns=4,
                visible=False
            )

            selfie_toggle = gr.State(value=False)
            with gr.Group(elem_classes=["selfie-section"]):
                selfie_button = gr.Button(
                    "📸 Add Selfie Verification",
                    variant="secondary",
                    elem_classes=["action-button", "selfie-button"]
                )
            with gr.Group(visible=False, elem_classes=["input-card"]) as selfie_group:
                with gr.Tabs():
                    with gr.TabItem("Capture Live Selfie"):
                        webcam_input = gr.Image(
                            label="Live Selfie",
                            sources=["webcam"],
                            interactive=True,
                            height=300,
                            elem_classes=["compact-input"]
                        )
                    with gr.TabItem("Upload Selfie"):
                        selfie_upload_input = gr.File(
                            label="Upload Selfie",
                            file_types=[".jpg", ".jpeg", ".png"],
                            file_count="single",
                            elem_classes=["compact-input", "file-input"]
                        )

            with gr.Row():
                process_button = gr.Button(
                    "Extract",
                    variant="primary",
                    elem_classes=["action-button", "extract-button"]
                )
                clear_button = gr.Button(
                    "Reset",
                    variant="secondary",
                    elem_classes=["action-button", "reset-button"]
                )

    # Main row for input and output (hidden initially)
    with gr.Row(variant="panel", equal_height=True, visible=False) as main_row:
        with gr.Column(scale=1, min_width=300):
            # Move input section here after extraction
            with gr.Group(elem_classes=["input-card"]):
                doc_type_moved = gr.Dropdown(
                    choices=["Citizenship", "License", "Passport"],
                    label="Document Type",
                    value="Citizenship",
                    elem_classes=["compact-input", "doc-type-dropdown"]
                )
                id_image_input_moved = gr.File(
                    label="Upload ID Images",
                    file_types=[".jpg", ".jpeg", ".png"],
                    file_count="multiple",
                    elem_classes=["compact-input", "file-input"]
                )
                image_preview_moved = gr.Gallery(
                    label="Uploaded Images Preview",
                    height="120px",
                    object_fit="cover",
                    elem_classes=["image-preview"],
                    columns=4,
                    visible=False
                )
                selfie_toggle_moved = gr.State(value=False)
                with gr.Group(elem_classes=["selfie-section"]):
                    selfie_button_moved = gr.Button(
                        "📸 Add Selfie Verification",
                        variant="secondary",
                        elem_classes=["action-button", "selfie-button"]
                    )
                with gr.Group(visible=False, elem_classes=["input-card"]) as selfie_group_moved:
                    with gr.Tabs():
                        with gr.TabItem("Capture Live Selfie"):
                            webcam_input_moved = gr.Image(
                                label="Live Selfie",
                                sources=["webcam"],
                                interactive=True,
                                height=300,
                                elem_classes=["compact-input"]
                            )
                        with gr.TabItem("Upload Selfie"):
                            selfie_upload_input_moved = gr.File(
                                label="Upload Selfie",
                                file_types=[".jpg", ".jpeg", ".png"],
                                file_count="single",
                                elem_classes=["compact-input", "file-input"]
                            )
                with gr.Row():
                    process_button_moved = gr.Button(
                        "Extract",
                        variant="primary",
                        elem_classes=["action-button", "extract-button"]
                    )
                    clear_button_moved = gr.Button(
                        "Reset",
                        variant="secondary",
                        elem_classes=["action-button", "reset-button"]
                    )

        with gr.Column(scale=2):
            with gr.Tabs(elem_classes=["output-tabs"]):
                with gr.Tab("Extracted Data"):
                    fields_output = gr.DataFrame(
                        label="Extracted Information",
                        headers=["Field", "Value", "Confidence"],
                        interactive=False,
                        elem_classes=["output-box"],
                        wrap=True
                    )
                with gr.Tab("Verification"):
                    verification_output = gr.DataFrame(
                        label="Verification Results",
                        headers=["Metric", "Value"],
                        interactive=False,
                        elem_classes=["output-box"],
                        wrap=True
                    )
            error_message = gr.Textbox(
                label="Error",
                lines=2,
                interactive=False,
                visible=False,
                elem_classes=["error-box"]
            )

    # Gallery section (hidden initially)
    gallery = gr.Gallery(
        label="Processed Images",
        height="300px",
        object_fit="contain",
        show_label=True,
        elem_classes=["image-gallery"],
        columns=4,
        visible=False
    )

    # Function to toggle selfie verification
    def toggle_selfie(current_state: bool, is_moved: bool) -> tuple:
        new_state = not current_state
        button_label = "📸 Remove Selfie Verification" if new_state else "📸 Add Selfie Verification"
        if is_moved:
            return (
                new_state, button_label, gr.update(visible=new_state),
                None, None, new_state, button_label, gr.update(visible=new_state)
            )
        return (
            new_state, button_label, gr.update(visible=new_state),
            None, None, new_state, button_label, gr.update(visible=new_state)
        )

    # Function to show image preview (fixed for both initial and moved states)
    def show_image_preview(id_images: List, is_moved: bool) -> tuple:
        logger.debug(f"Showing image preview, images: {id_images}, is_moved: {is_moved}")
        if not id_images or id_images == []:  # Explicitly hide if no images
            return gr.update(value=[], visible=False), gr.update(value=[], visible=False)
        image_paths = [img.name for img in id_images if img and hasattr(img, 'name')]
        if not image_paths:  # Additional check for empty paths
            return gr.update(value=[], visible=False), gr.update(value=[], visible=False)
        if is_moved:
            return gr.update(value=[], visible=False), gr.update(value=image_paths, visible=True)
        return gr.update(value=image_paths, visible=True), gr.update(value=[], visible=False)

    # Function to handle processing
    def handle_process(doc_type: str, id_images: List, webcam_image: Any, selfie_upload: Any, enable_selfie: bool) -> tuple:
        if not id_images:
            logger.warning("No ID images uploaded")
            return (
                gr.update(value=pd.DataFrame(), visible=True), gr.update(value=pd.DataFrame(), visible=True),
                gr.update(value=[], visible=True), gr.update(value="Please upload at least one ID image.", visible=True),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
                doc_type, id_images, enable_selfie, gr.update(value=[], visible=False)
            )

        id_image_paths = [img.name for img in id_images if img and hasattr(img, 'name')] if isinstance(id_images, list) else []
        if not id_image_paths:
            logger.warning("No valid ID image paths extracted")
            return (
                gr.update(value=pd.DataFrame(), visible=True), gr.update(value=pd.DataFrame(), visible=True),
                gr.update(value=[], visible=True), gr.update(value="Invalid ID image uploads.", visible=True),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
                doc_type, id_images, enable_selfie, gr.update(value=[], visible=False)
            )

        selfie_path = None
        if enable_selfie:
            if webcam_image and isinstance(webcam_image, str) and os.path.exists(webcam_image):
                selfie_path = webcam_image
            elif selfie_upload and hasattr(selfie_upload, 'name') and os.path.exists(selfie_upload.name):
                selfie_path = selfie_upload.name
            else:
                logger.warning("No valid selfie provided despite enable_selfie=True")
                return (
                    gr.update(value=pd.DataFrame(), visible=True), gr.update(value=pd.DataFrame(), visible=True),
                    gr.update(value=[], visible=True), gr.update(value="Please provide a valid selfie image.", visible=True),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
                    doc_type, id_images, enable_selfie, gr.update(value=[], visible=False)
                )

        logger.debug(f"Processing {len(id_image_paths)} ID images, selfie: {selfie_path}")
        result = process_images(doc_type, id_image_paths, selfie_path, enable_selfie)

        return (
            gr.update(value=result["fields_df"], visible=True),
            gr.update(value=result["verification_df"], visible=True),
            gr.update(value=result["gallery"], visible=True),
            gr.update(value=result["error"] or "", visible=bool(result["error"])),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
            doc_type, id_images, enable_selfie, gr.update(value=id_image_paths, visible=True)
        )

    # Function to clear all components
    def clear_all() -> tuple:
        logger.debug("Clearing all UI components")
        return (
            gr.update(value=pd.DataFrame(), visible=True),  # fields_output
            gr.update(value=pd.DataFrame(), visible=True),  # verification_output
            gr.update(value=[], visible=True),  # gallery
            gr.update(value="", visible=False),  # error_message
            False,  # selfie_toggle
            None,  # id_image_input
            None,  # webcam_input
            None,  # selfie_upload_input
            "📸 Add Selfie Verification",  # selfie_button
            gr.update(visible=False),  # selfie_group
            False,  # selfie_toggle_moved
            None,  # id_image_input_moved
            None,  # webcam_input_moved
            None,  # selfie_upload_input_moved
            "📸 Add Selfie Verification",  # selfie_button_moved
            gr.update(visible=False),  # selfie_group_moved
            gr.update(visible=False),  # main_row
            gr.update(visible=False),  # gallery
            gr.update(visible=True),  # input_column
            gr.update(value=[], visible=False),  # image_preview
            gr.update(value=[], visible=False),  # image_preview_moved
            "Citizenship",  # doc_type
            "Citizenship"  # doc_type_moved
        )

    # Event handlers
    selfie_button.click(
        fn=lambda state: toggle_selfie(state, False),
        inputs=[selfie_toggle],
        outputs=[selfie_toggle, selfie_button, selfie_group, webcam_input, selfie_upload_input,
                 selfie_toggle_moved, selfie_button_moved, selfie_group_moved]
    )

    selfie_button_moved.click(
        fn=lambda state: toggle_selfie(state, True),
        inputs=[selfie_toggle_moved],
        outputs=[selfie_toggle_moved, selfie_button_moved, selfie_group_moved, webcam_input_moved, selfie_upload_input_moved,
                 selfie_toggle, selfie_button, selfie_group]
    )

    # Ensure preview is hidden by default and only shows after upload
    id_image_input.upload(
        fn=lambda images: show_image_preview(images, False),
        inputs=[id_image_input],
        outputs=[image_preview, image_preview_moved]
    )

    id_image_input.clear(
        fn=lambda: show_image_preview([], False),
        inputs=[],
        outputs=[image_preview, image_preview_moved]
    )

    id_image_input_moved.upload(
        fn=lambda images: show_image_preview(images, True),
        inputs=[id_image_input_moved],
        outputs=[image_preview, image_preview_moved]
    )

    id_image_input_moved.clear(
        fn=lambda: show_image_preview([], True),
        inputs=[],
        outputs=[image_preview, image_preview_moved]
    )

    process_button.click(
        fn=handle_process,
        inputs=[doc_type, id_image_input, webcam_input, selfie_upload_input, selfie_toggle],
        outputs=[
            fields_output, verification_output, gallery, error_message,
            main_row, gallery, input_column,
            doc_type_moved, id_image_input_moved, selfie_toggle_moved, image_preview_moved
        ]
    )

    process_button_moved.click(
        fn=handle_process,
        inputs=[doc_type_moved, id_image_input_moved, webcam_input_moved, selfie_upload_input_moved, selfie_toggle_moved],
        outputs=[
            fields_output, verification_output, gallery, error_message,
            main_row, gallery, input_column,
            doc_type, id_image_input, selfie_toggle, image_preview
        ]
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            fields_output, verification_output, gallery, error_message,
            selfie_toggle, id_image_input, webcam_input, selfie_upload_input,
            selfie_button, selfie_group,
            selfie_toggle_moved, id_image_input_moved, webcam_input_moved, selfie_upload_input_moved,
            selfie_button_moved, selfie_group_moved,
            main_row, gallery, input_column,
            image_preview, image_preview_moved,
            doc_type, doc_type_moved
        ]
    )

    clear_button_moved.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            fields_output, verification_output, gallery, error_message,
            selfie_toggle, id_image_input, webcam_input, selfie_upload_input,
            selfie_button, selfie_group,
            selfie_toggle_moved, id_image_input_moved, webcam_input_moved, selfie_upload_input_moved,
            selfie_button_moved, selfie_group_moved,
            main_row, gallery, input_column,
            image_preview, image_preview_moved,
            doc_type, doc_type_moved
        ]
    )

if __name__ == "__main__":
    logger.info("Launching Gradio UI on http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)