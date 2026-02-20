import streamlit as st
import tempfile
import os
from src.extractor import extract_text_from_pdf
from src.embedding import SemanticMatcher
from src.ranking_engine import rank_resumes
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screening Dashboard")
st.write("Upload resumes + paste job description to rank candidates")
uploaded_files = st.file_uploader(
    "Upload resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)
job_description = st.text_area(
    "Paste Job Description",
    height=150
)
if st.button("🔍 Rank Candidates"):
    if not uploaded_files or not job_description:
        st.warning("Please upload resumes and enter a job description.")
        st.stop()

    matcher = SemanticMatcher()
    resume_texts = []
    resume_names = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        text = extract_text_from_pdf(tmp_path)
        resume_texts.append(text)
        resume_names.append(file.name)
        os.remove(tmp_path)

    job_embedding = matcher.model.encode(job_description)
    resume_embeddings = [
        matcher.model.encode(text) for text in resume_texts
    ]
    ranking_df = rank_resumes(
        job_embedding,
        resume_embeddings,
        resume_names
    )
    ranking_df["Match Percentage"] = (
        ranking_df["Similarity Score"] * 100
    ).round(2)
    st.success("Ranking complete!")
    st.dataframe(ranking_df)