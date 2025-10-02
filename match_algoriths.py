import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# --- Helper Functions ---
def extract_years_experience(text):
    """Extract years of experience (approx)"""
    matches = re.findall(r'(\d+)\s*\+?\s*year', text.lower())
    if matches:
        return max([int(m) for m in matches])  # take max if multiple
    return 0

def extract_education_level(text):
    """Simple education level detection"""
    text = text.lower()
    if "phd" in text or "doctorate" in text:
        return 3
    elif "master" in text or "msc" in text:
        return 2
    elif "bachelor" in text or "bsc" in text:
        return 1
    return 0

def skill_overlap(job_desc, resume, skills):
    """Count overlap of required skills"""
    job_tokens = set(job_desc.lower().split())
    resume_tokens = set(resume.lower().split())
    required = [s.lower() for s in skills]
    overlap = sum(1 for s in required if s in resume_tokens)
    return overlap / max(1, len(required))

# --- Main Ranking Function ---
def rank_resumes(job_desc, resumes, required_skills=None,
                 w_cos=0.35, w_bm25=0.35, w_skill=0.3):
    
    # === TF-IDF cosine similarities ===
    vectorizer = TfidfVectorizer(stop_words="english")
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

        # Years experience match
        cand_years = extract_years_experience(resume)
        year_score = 0
        

        # Final weighted score
        final_score = (
            w_cos * cos_sim +
            w_bm25 * bm25_n +
            w_skill * skill_score
        )

        results.append((resume, final_score, cos_sim, bm25_raw, skill_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results






# # --- Example Usage ---
# job_desc = "Looking for a machine learning engineer with Python, TensorFlow, and SQL experience. Minimum 3 years of experience. Master's degree preferred."
# resumes = [
#     "Worked 3 years as ML engineer, Python and TensorFlow expert, strong SQL background. MSc in Computer Science.",
#     "Software developer with C++ and Java, 5 years experience. Bachelor degree.",
#     "Data scientist skilled in Python, SQL, and deep learning with TensorFlow. 2 years industry experience. PhD in AI.",
#     "Frontend engineer, React and JavaScript, little data background. 4 years experience. Bachelor degree."
# ]

# from worker import extract_skill_from_jd_deepseek
# import json
# results=extract_skill_from_jd_deepseek(job_desc)
# results=json.loads(results)

# required_skills = results['skills']

# ranked = rank_resumes(job_desc, resumes, required_skills)

# for r in ranked:
#     print(f"Score={r[1]:.3f} | Cos={r[2]:.3f} | BM25_raw={r[3]:.2f} | Skills={r[4]:.2f} | Resume: {r[0]}")
