import streamlit as st
import pytesseract
import numpy as np
import cv2
import re
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_bytes
import time
import os

# Configure Tesseract path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_tesseract>'


# Preprocess function to optimize image size and quality for better OCR performance
def preprocess_image(image, max_size=(1000, 1000), enhance_contrast=True):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.thumbnail(max_size, Image.LANCZOS)

    # Convert image to grayscale and enhance contrast for better readability
    if enhance_contrast:
        image = image.convert('L')  # Convert to grayscale
        image = image.filter(ImageFilter.MedianFilter())  # Reduce noise
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Increase contrast level

    return image

def pdf_to_images(pdf_file, dpi=300):
    try:
        pages = convert_from_bytes(pdf_file, dpi=dpi)
        return pages
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return []

def clean_ocr_output(extracted_text):
    cleaned_text = extracted_text.replace("\n", " ")
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)
    cleaned_text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\s,.!?'\-।]+", "", cleaned_text)

    corrections = {
        'l ': ' ',
        'l': '',
        '।': '. ',
        'pleasel': 'please!',
        'quitel': 'quiet!',
    }
    for old, new in corrections.items():
        cleaned_text = cleaned_text.replace(old, new)

    sentences = re.split(r'(?<=[.!?]) ', cleaned_text)

    formatted_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'(?<=[\u0900-\u097F])(?=[a-zA-Z])', ' ', sentence)
        sentence = re.sub(r'(?<=[a-zA-Z])(?=[\u0900-\u097F])', ' ', sentence)
        formatted_sentences.append(sentence.capitalize())

    structured_text = '\n- '.join(formatted_sentences)
    return structured_text

def ocr_and_search(image, keyword):
    image = preprocess_image(image)
    image_cv = np.array(image)
    
    # Use Tesseract for both Hindi and English OCR
    start_time = time.time()
    
    # Extract text using Tesseract
    extracted_text_english = pytesseract.image_to_string(image_cv, lang='eng')
    extracted_text_hindi = pytesseract.image_to_string(image_cv, lang='hin')

    end_time = time.time()
    st.write(f"OCR completed in {end_time - start_time:.2f} seconds")

    extracted_text = f"{extracted_text_english}\n{extracted_text_hindi}"

    # Clean and structure OCR output
    structured_text = clean_ocr_output(extracted_text)

    # Highlight keyword in the output text
    if keyword:
        highlighted_text = re.sub(
            f"({keyword})",
            r"<span style='background-color: yellow; color: red; font-weight: bold;'>\1</span>",
            structured_text,
            flags=re.IGNORECASE
        )
    else:
        highlighted_text = structured_text

    return highlighted_text

# Streamlit app layout
st.title("OCR with Keyword Search")

# File uploader for both image and PDF files
uploaded_file = st.file_uploader("Upload an image or PDF (jpg, jpeg, png, pdf)", type=["jpg", "jpeg", "png", "pdf"])

# Text input for keyword
keyword = st.text_input("Enter a keyword to highlight:")

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension in [".jpg", ".jpeg", ".png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Run OCR"):
            with st.spinner('Processing the image, please wait...'):
                highlighted_text = ocr_and_search(image, keyword)
                st.markdown(highlighted_text, unsafe_allow_html=True)

    elif file_extension == ".pdf":
        with st.spinner('Converting PDF pages to images...'):
            pdf_images = pdf_to_images(uploaded_file.read())
            if pdf_images:
                st.write(f"Total pages extracted: {len(pdf_images)}")

        ocr_results = []
        if pdf_images and st.button("Run OCR on PDF"):
            for idx, page_image in enumerate(pdf_images):
                st.image(page_image, caption=f'Page {idx + 1}', use_column_width=True)

                with st.spinner(f'Processing page {idx + 1}...'):
                    page_text = ocr_and_search(page_image, keyword)
                    ocr_results.append(page_text)

            combined_text = "\n\n".join(ocr_results)
            st.markdown(combined_text, unsafe_allow_html=True)
