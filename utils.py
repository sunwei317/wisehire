import pandas as pd
import os
import fitz
from docx import Document



def save_or_append_excel(df: pd.DataFrame, file_path: str, sheet_name: str = 'Sheet1'):
    """
    Save a DataFrame to Excel or append it to an existing Excel file.

    Parameters:
        df (pd.DataFrame): DataFrame to save or append
        file_path (str): Path to the Excel file
        sheet_name (str): Excel sheet name (default 'Sheet1')
    """
    if not os.path.exists(file_path):
        # File does not exist, create it
        df.to_excel(file_path, index=False, sheet_name=sheet_name)
    else:
        # Append to existing file
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Get the next empty row in the sheet
            start_row = writer.sheets[sheet_name].max_row
            df.to_excel(writer, index=False, header=True, startrow=start_row, sheet_name=sheet_name)
            
# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_doc(file_path: str) -> str:
    """
    Extract text from .doc files
    You may need to install python-docx or other libraries
    """
    try:
        # For .doc files, you might need antiword or other tools
        # This is a placeholder implementation
        import subprocess
        result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            # Fallback: try with catdoc or other tools
            raise Exception("DOC file extraction failed")
    except Exception as e:
        raise Exception(f"Failed to extract text from DOC file: {str(e)}")
def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
