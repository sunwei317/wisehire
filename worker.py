from __future__ import annotations

import requests
import json
import logging
import os, mimetypes
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fpdf import FPDF
import fitz
from docx import Document

import redis
import requests
from celery import Celery
from celery.result import AsyncResult


from openai import OpenAI
import openai


from io import BytesIO
from mimetypes import guess_type
import random

from models import update_task

deepseek_base_url = "https://api.deepseek.com"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("celery_worker")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("ENV", "development")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    broker_url: str = os.getenv("CELERY_BROKER_URL", "") or os.getenv("BROKER_URL", "")
    backend_url: str = os.getenv("CELERY_BACKEND_URL", "") or os.getenv("BACKEND_URL", "")


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


SETTINGS = Settings()


if SETTINGS.env == "development":
    logger.info("Running in development mode")
    r = redis.Redis(host="localhost", port=6379, db=0)
    # Assuming default Celery queue name is 'celery'
    print("Clearing Redis queue 'celery' in development mode")
    r.delete("celery")  # Danger: Deletes the whole queue
else:
    logger.info("Running in production mode")
    r = redis.Redis(host="localhost", port=6379, db=0)
    # Assuming default Celery queue name is 'celery'
    r.delete("celery")  # Danger: Deletes the whole queue

# ---------------------------------------------------------------------------
# Redis (non‑destructive on import)
# ---------------------------------------------------------------------------
# Create a Redis client for occasional needs; do NOT mutate queues at import.
try:
    redis_client: Optional[redis.Redis] = redis.Redis.from_url(SETTINGS.redis_url)
except Exception as e:  # pragma: no cover
    logger.warning("Failed to initialize Redis client: %s", e)
    redis_client = None

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------
celery_app = Celery(
    "worker_development" if SETTINGS.env == "development" else "worker_production",
)

# In your Celery config or worker.py
celery_app.conf.broker_url = 'redis://localhost:6379/0'
celery_app.conf.result_backend = 'redis://localhost:6379/0'

queues = celery_app.control.inspect().active_queues()
from celery import current_app
from fastapi.responses import JSONResponse
from fastapi import  HTTPException


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


def extract_working_history_deepseek(resume_text: str):
    system_prompt = "You are an expert HR analyst. Given the plain text of a resume, you must analyze it and extract Work History. "

    prompt = f"""this is the resume, {resume_text}, generate the list of working history with title, company, location, 
    start and end dates, 
                responsibilities in valid JSON format."""

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",  # Use the latest model available
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content.strip()

    return result

def match_score(
    resume_text: str, working_history: str, job_description: str
) -> Dict[str, Any]:
    """
    Analyze candidate match using ChatGPT API

    Args:
        working_history: Candidate's work history text
        job_description: Job description text

    Returns:
        Dictionary containing analysis results
    """

    # Set up your OpenAI API key (you should set this as an environment variable)
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    example_response = {
        "name": "the candidate's full name from the resume",
        "overall_score": "int",
        "detailed_analysis": "string",
        "score_breakdown": {
            "string": {
                "score": "int",
                "max_score": "int",
                "explanation": "string",
            },
            "string": {
                "score": "int",
                "max_score": "int",
                "explanation": "string",
            },
            "string": {
                "score": "int",
                "max_score": "int",
                "explanation": "string",
            },
            "string": {
                "score": "int",
                "max_score": "int",
                "explanation": "string",
            },
        },
        "strengths": ["string", "...", "string"],
        "gaps": ["string", "...", "string"],
        "recommendation": ["string"],
        "recommendation_explanation": ["string"],
        "interview_questions": ["string", "...", "string"],
        "interview_focus": "string",
        "history": [
            {
                "company": "XYZ Corp",
                "status_during_tenure": "the company is ... ",
            }
        ],
    }

    # Create the prompt for the analysis
    prompt = f"""
    You are an expert HR analyst. Given a candidate's resume, working history and a job description,
    provide a matching score and detailed analysis.

    RESUME:
    {resume_text}

    WORKING HISTORY:
    {working_history}

    JOB DESCRIPTION:
    {job_description}

    Please provide:
    1. An overall match score between 0-100
    2. Detailed analysis of how the score is calculated
    3. Breakdown of scoring categories
    4. Strengths and relevant alignment
    5. Gaps and concerns
    6. Actionable recommendations
    7. Interview questions to ask to probe further
    8. Key focus areas for the interview
    9. Summary of company status during tenure and reason for leaving

    Instructions:
    - Be specific and data-backed in your analysis
    - Avoid generic statements

    根据不同领域的job description, 对候选人的不同方面的要求也是不一样的。仔细分析提供的job description, 对于不同的方面给出不同的权重来计算匹配成绩。

    Format your response as a JSON object with the following structure exactly, including the example values:
    {json.dumps(example_response, indent=2)}

    Ensure the response is valid JSON.
    """

    try:
        # Call the ChatGPT API
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper option
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR analyst with 20+ years of experience in candidate evaluation and job matching.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more deterministic output
            max_tokens=1500,
        )

        # Extract the response content
        analysis_text = response.choices[0].message.content.strip()

        # Try to parse the JSON response
        try:
            analysis_data = json.loads(analysis_text)
            return analysis_data
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            return {"raw_analysis": analysis_text}

    except Exception as e:
        return {"error": str(e)}


def analyze_working_history(work_history) -> dict:

    example_output={
    "company": "Standard BioTools Inc.",
    "tenure_period": "September 2023 to December 2023",
    "key_developments": [
                    {
                        "Financial Performance": [
                        "Reported a net loss of $20.997 million for Q3 2023, an improvement from the $29.426 million loss in the same period the previous year.",
                        "Revenue for Q3 2023 decreased by 1.918% to $25.15 million compared to $25.65 million in Q3 2022."
                        ]
                    },
                    {
                        "Product Development": [
                        "Continued revenue growth in proteomics (mass cytometry) following the launch of the Hyperion XTi Imaging System in April 2023.",
                        "Stabilized revenue in genomics (microfluidics) leading to a breakeven operating contribution on a year-to-date basis."
                        ]
                    },
                    {
                        "Corporate Strategy": [
                        "Announced an all-stock merger with SomaLogic on October 4, 2023, aiming to build a diversified leader in life sciences tools."
                        ]
                    }
                ]
  }

    input_text = f"""
                    Research the company {work_history}.
                    Do:
                    - Do a fast web search for the company details on web during the tenure, skip preview parsing. 
                    - Based on the company details, infer the company status during the tenure (e.g., fund raising, 
                    acquisition,startup, stable, expanding,re-organization).

                    Be analytical, avoid generalities, and ensure that each section supports
                    data-backed reasoning that could infer company status. No reference, url or citation is needed. Remove html tags.
                    generate the output as a list of company histories, and each company history is a valid JSON object.  no ```json.

                    Example output:
                    {json.dumps(example_output)}

                    """

    response = openai.responses.create(
        model="gpt-4o",  # "o3-deep-research",
        input=input_text,
        tools=[
            {"type": "web_search_preview"},
        ],
    )

    return response.output_text

@celery_app.task
def submit_pipeline(task_ids,user_id,resume_texts,job_description):

    print("start analysis, please wait")

    for (task_id,resume_text) in zip(task_ids,resume_texts):
        # 获取工作历史
        work_histories = extract_working_history_deepseek(resume_text)

        if isinstance(work_histories, str):
            work_histories = json.loads(work_histories)

        work_histories = work_histories.get("work_history", [])

        all_work_history_summary = []

        # 对每个雇主进行详细分析
        for work_history in work_histories:
            summary = analyze_working_history(work_history)
            all_work_history_summary.append(summary.replace('\"','"'))

        # 匹配评分
        match_score_result = match_score(
            resume_text, json.dumps(all_work_history_summary), job_description
        )

        if isinstance(match_score_result, str):
            try:
                match_score_result = json.loads(match_score_result)
            except json.JSONDecodeError as e:
                print("Failed to parse:", e)
                raise HTTPException(status_code=500, detail="Error parsing analysis result")
        try:
            # 构建分析结果
            analysis_result = {
                "full_name": match_score_result.get("name", "unknown"),
                "match_score": str(match_score_result["overall_score"]),
                "analysis_report": match_score_result["detailed_analysis"],
                "score_breakdown": match_score_result["score_breakdown"],
                "strengths": match_score_result["strengths"],
                "gaps": match_score_result["gaps"],
                "recommendations": match_score_result["recommendation"] + match_score_result["recommendation_explanation"],
                "interview_questions": match_score_result["interview_questions"],
                "interview_focus": match_score_result["interview_focus"],
                "detailed_work_histories": all_work_history_summary,
            }

            # 保存分析结果
            result_file_path = os.path.join("analysis_reports", f"task_id_{task_id}_user_{user_id}.json")
            with open(result_file_path, "w", encoding='utf-8') as _json:
                json.dump(analysis_result, _json, ensure_ascii=False, indent=4)

            # 更新任务状态
            update_data = {
                "task_id": task_id,
                "status": "completed",
                "result_json": result_file_path,
                "updated_at": datetime.now()
            }
            update_task(update_data)
        except Exception as e:
            print(e)
            update_data = {
                    "task_id": task_id,
                    "status": "failed",
                    "updated_at": datetime.now()
                }
            update_task(update_data)
            
    print("Analysis completed successfully!")

    return None