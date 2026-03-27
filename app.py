import os
import streamlit as st
import pandas as pd

from src.extractor import extract_pdf_text
from src.preprocessing import TextPreprocessor
from src.ner_extractor import SkillExtractor
from src.embedding import SemanticMatcher
from src.ranking_engine import rank_resumes


# ── Bug Fix 3: Cache the model so it loads only ONCE across all rerenders ─────
@st.cache_resource
def load_models():
    return SemanticMatcher(), TextPreprocessor(), SkillExtractor()


matcher, preprocessor, skill_extractor = load_models()

st.title("📄 NLP Resume Screener")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area("Paste Job Description Here")

if st.button("Rank Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        resume_embeddings = []
        resume_names = []
        resume_skills = []

        for file in uploaded_files:

            # ── Bug Fix 2: Pass file object directly — no temp file, no leak ──
            resume_text = extract_pdf_text(file)

            if not resume_text.strip():
                st.warning(f"{file.name} has no readable text.")
                continue

            # ── Bug Fix 4: Wire in preprocessing before embedding ─────────────
            cleaned_text = preprocessor.preprocess(resume_text)

            # ── Bug Fix 4: Wire in NER skill extraction ───────────────────────
            entities = skill_extractor.extract_entities(resume_text)
            skills_found = ", ".join(entities["skills"]) if entities["skills"] else "None detected"

            embedding = matcher.embed(cleaned_text)
            resume_embeddings.append(embedding)
            resume_names.append(file.name)
            resume_skills.append(skills_found)

        if not resume_embeddings:
            st.error("No valid resumes could be processed.")
        else:
            cleaned_jd = preprocessor.preprocess(job_description)
            job_embedding = matcher.embed(cleaned_jd)

            ranking = rank_resumes(job_embedding, resume_embeddings, resume_names)

            skills_map = dict(zip(resume_names, resume_skills))
            ranking["Matched Skills"] = ranking["Resume"].map(skills_map)

            st.subheader("📊 Resume Ranking")
            st.dataframe(ranking)