import os
import pdfplumber
from docx import Document

OUTPUT_DIR = "outputs/extracted_texts"


def extract_pdf_text(file_path):
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"[PDF ERROR] {file_path}: {e}")

    return text


def extract_docx_text(file_path):
    text = ""

    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    except Exception as e:
        print(f"[DOCX ERROR] {file_path}: {e}")

    return text


def save_text(filename, content):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(filename)[0] + ".txt"
    )

    try:
        with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(content)

    except Exception as e:
        print(f"[SAVE ERROR] {filename}: {e}")


def process_folder(folder_path):
    for file in os.listdir(folder_path):

        path = os.path.join(folder_path, file)

        if file.lower().endswith(".pdf"):
            text = extract_pdf_text(path)

        elif file.lower().endswith(".docx"):
            text = extract_docx_text(path)

        else:
            print(f"Skipping unsupported file: {file}")
            continue

        save_text(file, text)
        print(f"Processed: {file}")

import io
def extract_text_from_pdf(uploaded_file):
    """
    Used by Streamlit UI.
    Accepts uploaded PDF file object instead of file path.
    """
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"[UPLOAD PDF ERROR]: {e}")
    return text