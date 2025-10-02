import os
import json
import requests
from openai import OpenAI
from typing import Any, Dict

deepseek_base_url = "https://api.deepseek.com"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)

deepseek_model="deepseek-chat"


def analyze_working_history_deepseek(work_history) -> dict:

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
        请研究公司在{work_history}期间的情况。
        要求：
        - 快速搜索该公司在该期间的相关信息
        - 根据公司详情，推断该公司在该期间的状态（如融资、收购、初创、稳定、扩张、重组等）

        请确保分析具有数据支持，避免泛泛而谈，每个部分都应提供能够推断公司状态的数据支撑。
        不需要引用、URL或参考文献。移除HTML标签。

        重要：输出必须是一个严格的JSON对象，格式如下，不要包含任何其他文字或标记：

        {{
            "company": "公司名称",
            "tenure_period": "时间段",
            "key_developments": [
                {{
                    "分类名称": [
                        "具体发展1",
                        "具体发展2"
                    ]
                }}
            ]
        }}

        请严格按照上述JSON格式以英文输出，不要包含```json或其他任何前缀后缀。
        """
    url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deepseek_api_key}"
    }

    payload = {
        "model": "deepseek-chat",  # or "deepseek-reasoner"
        "messages": [
            {
                "role": "user", 
                "content": input_text
            }
        ],
        "stream": False,
        "web_search": True  # Enable web search functionality
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"



def match_score_only_deepseek(resume_text: str, job_description: str) -> Dict[str, Any]:
    
    # Create the prompt for the analysis
    prompt = f"""
    You are an expert HR analyst. Given a candidate's resume {resume_text}, and a job description {job_description},
    根据不同领域的job description, 对候选人的不同方面的要求也是不一样的。仔细分析提供的job description, 对于不同的方面给出不同的权重来计算匹配成绩。
    You must provide a matching score between 0 and 100, where 0 means no match at all and 100 means perfect match. If you don't know, the score is 0.
    Also extract the candidate's full name.
    The output is valid JSON format as {{"match_score":"int","full_name":""string}}. 
    """

    response = deepseek_client.chat.completions.create(
        model=deepseek_model,  # Use the latest model available
        messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR analyst with 20+ years of experience in candidate evaluation and job matching.",
                },
                {"role": "user", "content": prompt},
                
            ],
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=100,
            response_format={"type": "json_object"},
    )

    # Extract the response content
    match_score = response.choices[0].message.content.strip()
    
        # Try to parse the JSON response
    try:
        match_score = json.loads(match_score)
        return match_score
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw text
        return {"raw_analysis": match_score}


def match_score_from_deepseek(
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
    - if no full name is extract, just say 'unkown'.

    根据不同领域的job description, 对候选人的不同方面的要求也是不一样的。仔细分析提供的job description, 对于不同的方面给出不同的权重来计算匹配成绩。

    Format your response as a JSON object with the following structure exactly, including the example values:
    {json.dumps(example_response, indent=2)}

    Ensure the response is valid JSON.
    """

    response = deepseek_client.chat.completions.create(
        model=deepseek_model,  # Use the latest model available
        messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR analyst with 20+ years of experience in candidate evaluation and job matching.",
                },
                {"role": "user", "content": prompt},
                
            ],
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=1500,
            response_format={"type": "json_object"},
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



def extract_candidate_name_deepseek(resume_text: str):
    system_prompt = "You are an expert HR analyst.  "

    prompt = f"""Given the plain text of a resume, {resume_text}, you must analyze it and extract the full name of the candidate."""

    response = deepseek_client.chat.completions.create(
        model=deepseek_model,  # Use the latest model available
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    result = response.choices[0].message.content.strip()

    return result


def extract_working_history_deepseek(resume_text: str):
    system_prompt = "You are an expert HR analyst. Given the plain text of a resume, you must analyze it and extract Work History. "

    prompt = f"""this is the resume, {resume_text}, generate the list of working history with title, company, location, 
    start and end dates, 
                responsibilities in valid JSON format."""

    response = deepseek_client.chat.completions.create(
        model=deepseek_model,  # Use the latest model available
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

def extract_skill_from_jd_deepseek(job_description: str):
    system_prompt = "You are an expert HR analyst. Given the plain text of a job description, you must analyze it and extract required skills"

    prompt = f"""this is the job description, {job_description}, generate the list of job_description.
                
                Output in valid JSON format as the following format:
                {{
                "skills": ["string","string"],
                }}
                
                """

    response = deepseek_client.chat.completions.create(
        model=deepseek_model,  # Use the latest model available
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


def get_job_title_deepseek(job_descrition: str):
    system_prompt = "You are an expert HR analyst. Given the plain text of a job description, you must analyze it and extract the company and the job title. "

    prompt = f"""this is the job description, {job_descrition}, generate the job title and the company. output is the fromat 
                                        job_title|company
    """

    response = deepseek_client.chat.completions.create(
        model=deepseek_model,  # Use the latest model available
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,

    )

    result = response.choices[0].message.content.strip()

    return result

