"""
ATS Scorer — Multi-factor resume scoring system.

Score breakdown (total 100 points):
  Skill Match         40 pts  — JD skills found in resume
  Semantic Relevance  25 pts  — SBERT cosine similarity (rescaled)
  Experience Relevance 20 pts — Skills found in Experience/Internship/Projects
  Resume Structure    10 pts  — Key sections present in resume
  Keyword Coverage     5 pts  — TF-IDF overlap (rescaled)

Weights are based on the principle that demonstrated skill usage
(in context) is stronger evidence than listed keywords alone.
"""


def compute_ats_score(
    jd_skills: list,
    resume_skills: list,
    sbert_score: float,
    tfidf_score: float,
    skill_locations: dict,
    detected_sections: list,
) -> dict:
    """
    Compute a multi-factor ATS score for one resume against a JD.

    Args:
        jd_skills:         skills extracted from the job description
        resume_skills:     skills extracted from the resume
        sbert_score:       cosine similarity from SBERT (0.0 - 1.0)
        tfidf_score:       cosine similarity from TF-IDF (0.0 - 1.0)
        skill_locations:   {skill_name -> section_name} from section parser
        detected_sections: list of section names found in resume

    Returns:
        {
            "total":               84,
            "skill_match":         32,
            "semantic_relevance":  21,
            "experience_relevance":18,
            "resume_structure":     8,
            "keyword_coverage":     5,
            "breakdown": {
                "skill_match_pct":        0.80,
                "matched_skills":         [...],
                "missing_skills":         [...],
                "skills_in_experience":   [...],
                "skills_only_in_skills":  [...],
                "sections_found":         [...],
                "sections_missing":       [...],
            }
        }
    """

    # ── 1. Skill Match (40 pts) ───────────────────────────────────────────────
    if jd_skills:
        resume_skill_set = set(resume_skills)
        jd_skill_set = set(jd_skills)
        matched = sorted(jd_skill_set & resume_skill_set)
        missing = sorted(jd_skill_set - resume_skill_set)
        skill_match_pct = len(matched) / len(jd_skill_set)
    else:
        matched = []
        missing = []
        skill_match_pct = 0.0

    skill_match_score = round(skill_match_pct * 40)

    # ── 2. Semantic Relevance (25 pts) ────────────────────────────────────────
    # SBERT scores for resume-JD pairs typically range 0.2 - 0.8.
    # We normalize to 0-1 using a practical range clip, then scale to 25.
    sbert_normalized = min(max((sbert_score - 0.15) / 0.65, 0.0), 1.0)
    semantic_score = round(sbert_normalized * 25)

    # ── 3. Experience Relevance (20 pts) ──────────────────────────────────────
    # Skills found in Experience/Internship/Projects = demonstrated usage.
    # Skills found only in Skills section = listed but undemonstrated.
    EXPERIENCE_SECTIONS = {"Experience", "Internship", "Projects", "Research"}

    skills_in_experience = [
        s for s in matched
        if skill_locations.get(s, "") in EXPERIENCE_SECTIONS
    ]
    skills_only_in_skills = [
        s for s in matched
        if skill_locations.get(s, "") == "Skills"
    ]

    if matched:
        experience_ratio = len(skills_in_experience) / len(matched)
    else:
        experience_ratio = 0.0

    experience_score = round(experience_ratio * 20)

    # ── 4. Resume Structure (10 pts) ──────────────────────────────────────────
    # Award points for having key sections present.
    # Max 2 pts per section, up to 10 pts total.
    IMPORTANT_SECTIONS = {
        "Experience": 3,
        "Skills":     3,
        "Education":  2,
        "Projects":   2,
    }

    structure_score = 0
    sections_found = []
    sections_missing = []

    for section, pts in IMPORTANT_SECTIONS.items():
        if section in detected_sections:
            structure_score += pts
            sections_found.append(section)
        else:
            sections_missing.append(section)

    structure_score = min(structure_score, 10)

    # ── 5. Keyword Coverage (5 pts) ───────────────────────────────────────────
    # TF-IDF scores typically range 0.0 - 0.4 for resume-JD pairs.
    tfidf_normalized = min(tfidf_score / 0.4, 1.0)
    keyword_score = round(tfidf_normalized * 5)

    # ── Total ─────────────────────────────────────────────────────────────────
    total = (
        skill_match_score
        + semantic_score
        + experience_score
        + structure_score
        + keyword_score
    )
    total = min(total, 100)  # cap at 100

    return {
        "total":                total,
        "skill_match":          skill_match_score,
        "semantic_relevance":   semantic_score,
        "experience_relevance": experience_score,
        "resume_structure":     structure_score,
        "keyword_coverage":     keyword_score,
        "breakdown": {
            "skill_match_pct":       round(skill_match_pct * 100),
            "matched_skills":        matched,
            "missing_skills":        missing,
            "skills_in_experience":  skills_in_experience,
            "skills_only_in_skills": skills_only_in_skills,
            "sections_found":        sections_found,
            "sections_missing":      sections_missing,
        }
    }


def ats_grade(total: int) -> tuple:
    """
    Convert ATS score to a letter grade and label.

    Returns (grade, label, color)
    """
    if total >= 80:
        return "A", "Excellent Match", "green"
    elif total >= 65:
        return "B", "Good Match", "blue"
    elif total >= 50:
        return "C", "Moderate Match", "orange"
    elif total >= 35:
        return "D", "Weak Match", "red"
    else:
        return "F", "Poor Match", "red"