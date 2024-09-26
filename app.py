import gradio as gr
import pytesseract
from PIL import Image
import re
import os

# Set the path to the Tesseract executable (Windows Only)
# Uncomment and update the path if using Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function for OCR and keyword search
def ocr_and_search(image, keyword):
    # Perform OCR on the uploaded image
    extracted_text = pytesseract.image_to_string(image, lang='eng+hin')

    # Perform a simple keyword search
    if keyword:
        highlighted_text = re.sub(f"({keyword})", r"**\1**", extracted_text, flags=re.IGNORECASE)
    else:
        highlighted_text = extracted_text

    return highlighted_text


# Create the Gradio Interface with updated components
gr.Interface(
    fn=ocr_and_search,
    inputs=[gr.components.Image(type="pil"), gr.components.Textbox()],
    outputs=gr.components.Textbox(),
    title="OCR with Keyword Search",
    description="Upload an image with text in Hindi and English, and search for a keyword in the extracted text."
    
).launch()
