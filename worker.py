from __future__ import annotations

import json
import logging
import os, mimetypes
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import redis
import requests
from celery import Celery
from celery.result import AsyncResult
from match_refined_algorithms import rank_resumes_enhanced
from utils import (
    save_or_append_excel,
)
from openai import OpenAI
import openai

import pandas as pd
from io import BytesIO
from mimetypes import guess_type
import random

from deepseek_utils import (
    match_score_from_deepseek,
    match_score_only_deepseek,
    extract_skill_from_jd_deepseek,
    extract_working_history_deepseek,
    analyze_working_history_deepseek    
)

from openai_utils import (
    match_score
)
from models import update_task


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


@celery_app.task
def submit_pipeline(task_ids,user_id,resume_texts,job_description,score_threshold=10):

    print("start analysis, please wait")
    
    scores_deepseek=[]

    scores_standard=[]
    full_names=[]
    jobs=[]
    jobs.append(job_description[:50])
    for (task_id,resume_text) in zip(task_ids,resume_texts):

        match_score_deepeeek=match_score_only_deepseek(resume_text,job_description)
        scores_deepseek.append(match_score_deepeeek.get("match_score","0"))

        # if float(match_score.get("match_score","0"))<score_threshold:
        #          # 更新任务状态
        #         update_data = {
        #             "task_id": task_id,
        #             "status": "completed",
        #             "score": match_score.get("match_score","0"),
        #             "candidate": match_score.get("full_name","unkown"),
        #             "result_json": "",
        #             "updated_at": datetime.now()
        #         }
        #         update_task(update_data)

        
        skills=extract_skill_from_jd_deepseek(job_description)
        skills=json.loads(skills)
        required_skills = skills['skills']

        # full_name=extract_candidate_name_deepseek(resume_text[:100])
        
        print("=== Enhanced Resume Ranking with Project Matching ===")
        match_score_standard= rank_resumes_enhanced(job_description, [resume_text], required_skills)[0]

        full_names.append("full_name")
        scores_standard.append(match_score_standard)

        if float(match_score_standard)<score_threshold:
                 # 更新任务状态
                update_data = {
                    "task_id": task_id,
                    "status": "completed",
                    "score": match_score_standard,
                    "candidate": "full_name",
                    "result_json": "",
                    "updated_at": datetime.now()
                }

                print(update_data)

                update_task(update_data)
        
        else:
            # 获取工作历史
            work_histories = extract_working_history_deepseek(resume_text)

            if isinstance(work_histories, str):
                work_histories = json.loads(work_histories)

            work_histories = work_histories.get("work_history", [])

            all_work_history_summary = []

            # 对每个雇主进行详细分析
            for work_history in work_histories:
                # summary = analyze_working_history(work_history)
                summary = analyze_working_history_deepseek(work_history)
                all_work_history_summary.append(summary.replace('\"','"'))

            # 匹配评分
            match_score_result = match_score_from_deepseek(
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
                    "score": analysis_result["match_score"],
                    "candidate": analysis_result["full_name"] if len(analysis_result["full_name"])>18 else "unknown",
                    "result_json": result_file_path,
                    "updated_at": datetime.now()
                }
                print(len(analysis_result["full_name"]))
                print(update_data)

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

    scores=pd.DataFrame({"full_name":full_names,"socre_deepseek":scores_deepseek,"score_standard":scores_standard})

    save_or_append_excel(scores,"score_comparation.xlsx")

    return None