import streamlit as st
import pandas as pd
import tempfile
from src.extractor import extract_pdf_text
from src.embedding import SemanticMatcher
from src.ranking_engine import rank_resumes
st.title("📄 NLP Resume Screener")
matcher = SemanticMatcher()
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
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_path = tmp.name

            resume_text = extract_pdf_text(temp_path)
            if len(resume_text.strip()) == 0:
                st.warning(f"{file.name} has no readable text.")
                continue

            embedding = matcher.embed(resume_text)
            resume_embeddings.append(embedding)
            resume_names.append(file.name)

        job_embedding = matcher.embed(job_description)
        ranking = rank_resumes(
            job_embedding,
            resume_embeddings,
            resume_names
        )
        st.subheader("📊 Resume Ranking")
        st.dataframe(ranking)
