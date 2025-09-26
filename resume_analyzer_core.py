import io
from fastapi import FastAPI, File, UploadFile, Response
import fitz  # PyMuPDF for text extraction
import openai
from fpdf import FPDF
from docx import Document
import os
import json
from typing import Dict, Any
from openai import OpenAI

app = FastAPI()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or hardcode for testing

deepseek_base_url = "https://api.deepseek.com"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)

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


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def extract_working_history_chatgpt(resume_text: str) -> dict:

    system_prompt = "You are an expert HR analyst. Given the plain text of a resume, you must analyze it and extract Work History. "

    prompt = f"""this is the resume, {resume_text}, generate the list of working history with title, company, location, start and end dates, 
                responsibilities in JSON format."""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content.strip()

    return result









# Create a simple PDF report from ChatGPT's analysis
def create_pdf_report_from_text(report_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in report_text.splitlines():
        pdf.multi_cell(0, 10, line.strip())

    # ✅ 输出为 string，然后安全编码为 PDF 字节
    pdf_str = pdf.output(dest="S")
    return pdf_str.encode("latin-1", errors="ignore")


def normalize_text(text: str) -> str:
    replacements = {
        "–": "-",
        "—": "-",  # en dash/em dash to hyphen
        "“": '"',
        "”": '"',  # curly quotes to straight
        "‘": "'",
        "’": "'",  # smart apostrophes to '
        "…": "...",  # ellipsis to three dots
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

