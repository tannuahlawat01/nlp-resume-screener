import os
import io
import pdfplumber
from docx import Document

OUTPUT_DIR = "outputs/extracted_texts"

def extract_pdf_text(source):
    """
    Extract text from a PDF file.
    Args:
        source: Either a file path (str) or a Streamlit UploadedFile / file-like object.
    Returns:
        Extracted text as a string.
    """
    text = ""
    try:
        # If source is a string treat it as a file path, otherwise wrap in BytesIO
        if isinstance(source, str):
            pdf_source = source
        else:
            pdf_source = io.BytesIO(source.read())
        with pdfplumber.open(pdf_source) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"[PDF ERROR] {source}: {e}")
    return text

def extract_docx_text(file_path):
    """
    Extract text from a DOCX file.
    Args:
        file_path: Path to the .docx file.
    Returns:
        Extracted text as a string.
    """
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"[DOCX ERROR] {file_path}: {e}")

    return text

def save_text(filename, content):
    """Save extracted text to the outputs directory."""
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
    """Process all PDF and DOCX resumes in a folder and save extracted text."""
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
