import json
import openai
from typing import Any, Dict



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


