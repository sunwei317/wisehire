import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


# --- Enhanced Helper Functions ---
def extract_years_experience(text):
    """Enhanced years of experience extraction with more patterns"""
    patterns = [
        r'(\d+)\s*\+?\s*year',
        r'(\d+)\s*\+?\s*yr',
        r'experience.*?(\d+)\s*year',
        r'(\d+)\s*years?\s*experience'
    ]
    
    max_years = 0
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            max_years = max(max_years, max([int(m) for m in matches]))
    return max_years

def extract_education_level(text):
    """Enhanced education level detection with more degrees"""
    text = text.lower()
    education_keywords = {
        'phd': 4, 'doctorate': 4, 'doctoral': 4,
        'master': 3, 'msc': 3, 'm.s.': 3, 'mba': 3,
        'bachelor': 2, 'bsc': 2, 'b.s.': 2, 'undergraduate': 2,
        'associate': 1, 'diploma': 1, 'certificate': 1
    }
    
    for keyword, score in education_keywords.items():
        if keyword in text:
            return score
    return 0

def skill_overlap(job_desc, resume, skills):
    """Enhanced skill matching with synonyms and levels"""
    skill_synonyms = {
        'python': ['python', 'py'],
        'tensorflow': ['tensorflow', 'tf'],
        'sql': ['sql', 'mysql', 'postgresql'],
        'machine learning': ['machine learning', 'ml', 'ai'],
        'deep learning': ['deep learning', 'dl'],
        'react': ['react', 'react.js', 'reactjs'],
        'javascript': ['javascript', 'js']
    }
    
    job_tokens = set(job_desc.lower().split())
    resume_tokens = set(resume.lower().split())
    
    matched_skills = 0
    for skill in skills:
        skill_variants = skill_synonyms.get(skill.lower(), [skill.lower()])
        if any(variant in resume_tokens for variant in skill_variants):
            matched_skills += 1
    
    return matched_skills / max(1, len(skills))

def extract_project_experience(text, job_keywords):
    """Extract and score project experience relevance"""
    # Project indicators
    project_indicators = [
        r'project', r'developed', r'built', r'created', r'implemented',
        r'led', r'managed', r'designed', r'architected'
    ]
    
    # Count project mentions
    project_count = 0
    for indicator in project_indicators:
        project_count += len(re.findall(indicator, text.lower()))
    
    # Check for job-relevant technologies in project context
    relevant_tech_count = 0
    for keyword in job_keywords:
        # Look for keywords near project indicators (simplified)
        if keyword.lower() in text.lower():
            relevant_tech_count += 1
    
    # Calculate scores
    project_density = min(project_count / 5, 1.0)  # Normalize project count
    tech_relevance = min(relevant_tech_count / max(1, len(job_keywords)), 1.0)
    
    # Combined score (weighted average)
    return 0.6 * tech_relevance + 0.4 * project_density

def extract_achievements(text):
    """Detect achievement indicators"""
    achievement_indicators = [
        r'increased', r'improved', r'reduced', r'led',
        r'managed', r'developed', r'created', r'achieved',
        r'awarded', r'promoted', r'optimized', r'scaled'
    ]
    
    achievement_count = 0
    for indicator in achievement_indicators:
        achievement_count += len(re.findall(indicator, text.lower()))
    
    return min(achievement_count / 5, 1.0)  # Normalize

def extract_job_stability(text):
    """Assess job stability based on tenure patterns"""
    # Simple version: count long tenure mentions
    tenure_patterns = [
        r'(\d+)\+?\s*years?\s+at',
        r'(\d+)\+?\s*years?\s+tenure',
        r'long.?term'
    ]
    
    stability_score = 0
    for pattern in tenure_patterns:
        if re.search(pattern, text.lower()):
            stability_score += 0.3
    
    return min(stability_score, 1.0)

def calculate_resume_quality(text):
    """Assess resume structure and completeness"""
    sections = ['experience', 'education', 'skills', 'projects']
    section_count = sum(1 for section in sections if section in text.lower())
    
    # Check for quantifiable achievements
    quant_pattern = r'\d+%|\$\d+|\d+\s*(?:years?|months?)'
    quant_count = len(re.findall(quant_pattern, text.lower()))
    
    structure_score = section_count / len(sections)
    quant_score = min(quant_count / 5, 1.0)
    
    return (structure_score + quant_score) / 2

def extract_job_keywords(job_desc):
    """Extract relevant keywords from job description"""
    # Simple keyword extraction - can be enhanced with NLP
    technical_keywords = [
        'python', 'java', 'javascript', 'sql', 'tensorflow', 
        'react', 'machine learning', 'deep learning', 'aws',
        'docker', 'kubernetes', 'api', 'database', 'cloud'
    ]
    
    found_keywords = []
    for keyword in technical_keywords:
        if keyword in job_desc.lower():
            found_keywords.append(keyword)
    
    return found_keywords

# --- Enhanced Main Ranking Function ---
def rank_resumes_enhanced(job_desc, resumes, required_skills=None, 
                         w_cos=0.25, w_bm25=0.25, w_skill=0.25, 
                         w_project=0.1, w_achieve=0.05, w_stability=0.05, w_quality=0.05):
    
    # Validate weights sum to 1
    weights = [w_cos, w_bm25, w_skill,  w_project, w_achieve, w_stability, w_quality]
    if abs(sum(weights) - 1.0) > 0.01:
        raise ValueError("Weights must sum to 1.0")
    
    # Extract job keywords for project matching
    job_keywords = extract_job_keywords(job_desc)
    if required_skills:
        job_keywords.extend([skill.lower() for skill in required_skills])
    
    # === TF-IDF cosine similarities ===
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([job_desc] + resumes)
    cosine_sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # === BM25 scores ===
    tokenized_job = [job_desc.lower().split()]
    bm25 = BM25Okapi(tokenized_job)
    bm25_scores = [bm25.get_scores(resume.lower().split())[0] for resume in resumes]
    bm25_norm = [1 / (1 + np.exp(-0.1 * (s - 5))) for s in bm25_scores]

    results = []
    for resume, cos_sim, bm25_raw, bm25_n in zip(resumes, cosine_sims, bm25_scores, bm25_norm):
        # Skill overlap
        skill_score = skill_overlap(job_desc, resume, required_skills) if required_skills else 0
        
        # Project experience match
        project_score = extract_project_experience(resume, job_keywords)
        
        # Other factors
        achieve_score = extract_achievements(resume)
        stability_score = extract_job_stability(resume)
        quality_score = calculate_resume_quality(resume)

        # Final weighted score
        final_score = (
            w_cos * cos_sim +
            w_bm25 * bm25_n +
            w_skill * skill_score +
            w_project * project_score +
            w_achieve * achieve_score +
            w_stability * stability_score +
            w_quality * quality_score
        )
        results.append(round(float(final_score),2)*100)
    return results

# # --- Example Usage ---
# job_desc = "Looking for a machine learning engineer with Python, TensorFlow, and SQL experience. Minimum 3 years of experience. Master's degree preferred. Experience with end-to-end ML projects required."
# resumes = [
#     "Worked 3 years as ML engineer. Developed multiple machine learning projects using Python and TensorFlow. Built SQL databases for data storage. MSc in Computer Science. Led a team of 3 developers on a recommendation system project.",
#     "Software developer with C++ and Java, 5 years experience. Bachelor degree. Worked on several backend projects but limited ML experience.",
#     "Data scientist skilled in Python, SQL, and deep learning with TensorFlow. 2 years industry experience. PhD in AI. Implemented end-to-end ML pipeline project that improved accuracy by 20%. Developed real-time prediction system.",
#     "Frontend engineer, React and JavaScript, little data background. 4 years experience. Bachelor degree. 2+ years at current company. Built several web applications but no ML projects."
# ]
# from worker import extract_skill_from_jd_deepseek
# import json
# results=extract_skill_from_jd_deepseek(job_desc)
# results=json.loads(results)

# required_skills = results['skills']

# print("=== Enhanced Resume Ranking with Project Matching ===")
# ranked = rank_resumes_enhanced(job_desc, resumes, required_skills)

# print(ranked)