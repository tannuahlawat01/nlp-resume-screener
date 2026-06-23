import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pandas as pd
from datetime import datetime

from src.extractor import extract_pdf_text
from src.preprocessing import TextPreprocessor
from src.ner_extractor import SkillExtractor
from src.embedding import SemanticMatcher
from src.tfidf_matcher import TFIDFMatcher
from src.ranking_engine import rank_resumes
from src.section_parser import parse_sections, locate_skills_in_sections


@st.cache_resource
def load_models():
    return SemanticMatcher(), TextPreprocessor(), SkillExtractor(), TFIDFMatcher()


matcher, preprocessor, skill_extractor, tfidf_matcher = load_models()


def match_label(score: float) -> tuple:
    if score >= 0.55:
        return "🟢 Strong Match", "green"
    elif score >= 0.35:
        return "🟡 Moderate Match", "orange"
    else:
        return "🔴 Weak Match", "red"


def match_interpretation(score: float) -> str:
    """Feature 2: Human-readable score interpretation."""
    if score >= 0.55:
        return (
            "✅ Strong semantic match with the job description. "
            "This candidate covers most required skills and context. "
            "Recommended for shortlisting."
        )
    elif score >= 0.35:
        return (
            "⚠️ Partial match. The candidate has relevant experience "
            "but is missing some key skills. Review the skill breakdown "
            "before deciding."
        )
    else:
        return (
            "❌ Weak match. This resume does not align well with the "
            "job description. Significant skill gaps detected."
        )


def improvement_suggestions(missing: list) -> list:
    """Feature 1: Actionable improvement suggestions for missing skills."""
    suggestions = []
    for skill in missing:
        skill_lower = skill.lower()

        if any(x in skill_lower for x in ["docker", "kubernetes", "k8s"]):
            suggestions.append(
                f"**{skill}** — Add a project or experience where you containerised "
                f"an application. Even a personal project using Docker Compose counts."
            )
        elif any(x in skill_lower for x in ["aws", "gcp", "azure", "cloud"]):
            suggestions.append(
                f"**{skill}** — Consider getting a free-tier account and deploying "
                f"a project. Mention it in your Projects section with the service used."
            )
        elif any(x in skill_lower for x in ["pytorch", "tensorflow", "keras"]):
            suggestions.append(
                f"**{skill}** — Build a small ML project (image classifier, text "
                f"classifier) and mention the framework explicitly in your resume."
            )
        elif any(x in skill_lower for x in ["sql", "postgresql", "mysql", "mongodb"]):
            suggestions.append(
                f"**{skill}** — If you've used any database in a project, name it "
                f"explicitly. Recruiters scan for database keywords."
            )
        elif any(x in skill_lower for x in ["react", "node", "javascript", "typescript"]):
            suggestions.append(
                f"**{skill}** — Add to your Skills section and mention a project "
                f"where you used it, even if it was small."
            )
        elif any(x in skill_lower for x in ["ci/cd", "github actions", "jenkins"]):
            suggestions.append(
                f"**{skill}** — If your projects are on GitHub, set up a basic "
                f"GitHub Actions workflow. It's a 30-minute task that adds a "
                f"strong keyword to your resume."
            )
        elif any(x in skill_lower for x in ["machine learning", "deep learning", "nlp"]):
            suggestions.append(
                f"**{skill}** — Make sure your ML projects are described with "
                f"technical specifics — model type, dataset, metric. Vague "
                f"descriptions don't register as ML experience."
            )
        else:
            suggestions.append(
                f"**{skill}** — If you have any experience with this (coursework, "
                f"projects, self-study), add it explicitly to your Skills or "
                f"Projects section."
            )
    return suggestions


def section_tips(locations: dict, matched: list) -> list:
    """Feature 3: Tips based on where skills appear in the resume."""
    tips = []
    skills_only_in_skills_section = [
        skill for skill in matched
        if locations.get(skill, "") == "Skills"
    ]
    skills_in_experience = [
        skill for skill in matched
        if locations.get(skill, "") in ["Experience", "Internship", "Projects"]
    ]

    if skills_only_in_skills_section:
        skill_list = ", ".join(skills_only_in_skills_section[:4])
        tips.append(
            f"⚠️ **{skill_list}** appear only in your Skills section. "
            f"Mentioning them with context in your Experience or Projects "
            f"section significantly strengthens your resume — recruiters "
            f"trust demonstrated usage over listed skills."
        )

    if len(skills_in_experience) == 0 and matched:
        tips.append(
            "⚠️ None of your matched skills appear in your Experience section. "
            "Try weaving key technologies into your bullet points "
            "(e.g. 'Built REST API using FastAPI and PostgreSQL')."
        )

    if not tips:
        tips.append(
            "✅ Your skills are well-distributed across resume sections. "
            "Good structure for ATS and human reviewers."
        )

    return tips


def generate_report(
    resume_name: str,
    jd_text: str,
    jd_skills: list,
    score: float,
    matched: list,
    missing: list,
    locations: dict,
    suggestions: list,
    tips: list
) -> str:
    """Feature 4: Generate plain text report for download."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    label, _ = match_label(score)
    interpretation = match_interpretation(score)

    lines = [
        "=" * 60,
        "NLP RESUME SCREENER — MATCH ANALYSIS REPORT",
        f"Generated: {now}",
        "=" * 60,
        "",
        f"Resume:         {resume_name}",
        f"Match Score:    {int(score * 100)}%  {label}",
        "",
        f"Assessment: {interpretation}",
        "",
        "-" * 60,
        "JOB DESCRIPTION SKILLS DETECTED",
        "-" * 60,
        ", ".join(jd_skills) if jd_skills else "None detected",
        "",
        "-" * 60,
        f"MATCHED SKILLS ({len(matched)}/{len(jd_skills) if jd_skills else 0})",
        "-" * 60,
    ]

    for skill in matched:
        section = locations.get(skill, "Unknown")
        lines.append(f"  ✅ {skill:25s} [Found in: {section}]")

    lines += [
        "",
        "-" * 60,
        f"MISSING SKILLS ({len(missing)}/{len(jd_skills) if jd_skills else 0})",
        "-" * 60,
    ]
    for skill in missing:
        lines.append(f"  ❌ {skill}")

    lines += [
        "",
        "-" * 60,
        "IMPROVEMENT SUGGESTIONS",
        "-" * 60,
    ]
    for i, s in enumerate(suggestions, 1):
        clean = s.replace("**", "")
        lines.append(f"{i}. {clean}")

    lines += [
        "",
        "-" * 60,
        "SECTION TIPS",
        "-" * 60,
    ]
    for tip in tips:
        clean = tip.replace("**", "").replace("⚠️", "[!]").replace("✅", "[OK]")
        lines.append(f"• {clean}")

    lines += ["", "=" * 60, "End of Report", "=" * 60]
    return "\n".join(lines)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 About This Tool")
    st.markdown(
        "This tool helps **recruiters and job seekers** analyse how well "
        "a resume matches a job description using NLP.\n\n"
        "It combines **Sentence-BERT** (semantic understanding) with "
        "**TF-IDF** (keyword matching) for a hybrid relevance score."
    )

    st.divider()
    st.markdown("## ⚙️ How It Works")
    st.markdown(
        "1. **Upload** resume PDF(s)\n"
        "2. **Paste** the job description\n"
        "3. **Click** Rank Resumes\n\n"
        "The app extracts skills, computes semantic similarity, "
        "ranks candidates, and shows exactly which skills matched, "
        "which are missing, and how to improve."
    )

    st.divider()
    st.markdown("## 📈 Model Benchmark")
    st.caption("SBERT vs TF-IDF on 3 labeled JD-resume pairs.")

    eval_path = os.path.join("evaluation", "results", "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        avg = next(r for r in eval_data if r["JD"] == "AVERAGE")
        col1, col2 = st.columns(2)
        col1.metric(
            "SBERT P@1",
            f"{float(avg['SBERT P@1']):.0%}",
            f"+{(float(avg['SBERT P@1']) - float(avg['TFIDF P@1'])):.0%} vs TF-IDF"
        )
        col2.metric("TF-IDF P@1", f"{float(avg['TFIDF P@1']):.0%}")
        rows = [r for r in eval_data if r["JD"] != "AVERAGE"]
        chart_df = pd.DataFrame({
            "JD":         [r["JD"] for r in rows],
            "SBERT P@1":  [r["SBERT P@1"] for r in rows],
            "TF-IDF P@1": [r["TFIDF P@1"] for r in rows],
        }).set_index("JD")
        st.bar_chart(chart_df, use_container_width=True)
    else:
        st.info("Run `python -m evaluation.evaluate` to generate benchmark.")

    st.divider()
    st.markdown("## 🏷️ Score Guide")
    st.markdown(
        "🟢 **Strong Match** — Score ≥ 55%\n\n"
        "🟡 **Moderate Match** — Score 35–55%\n\n"
        "🔴 **Weak Match** — Score < 35%"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📄 NLP Resume Screener")
st.markdown(
    "#### Rank candidates and improve resumes using semantic AI — "
    "not just keyword matching."
)
st.info(
    "💡 **Recruiters:** Upload multiple resumes to rank candidates. "
    "**Job seekers:** Upload your resume to see how well it matches a JD "
    "and get specific improvement tips.",
    icon="💡"
)
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📁 Upload Resume(s)")
    st.caption("PDF format only. Upload one (job seeker) or many (recruiter).")
    uploaded_files = st.file_uploader(
        "Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) ready.")

with col_right:
    st.markdown("### 📋 Job Description")
    st.caption("Paste the full JD including required and nice-to-have skills.")
    job_description = st.text_area(
        "JD",
        height=220,
        placeholder="e.g. We are looking for a Python backend engineer with FastAPI, PostgreSQL, Docker and AWS experience...",
        label_visibility="collapsed"
    )

st.divider()

run_col, _ = st.columns([1, 3])
with run_col:
    run_button = st.button(
        "🔍 Rank Resumes", type="primary", use_container_width=True
    )

# ── Processing ────────────────────────────────────────────────────────────────
if run_button:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one resume PDF.")
    elif not job_description.strip():
        st.warning("⚠️ Please paste a job description.")
    else:
        jd_entities = skill_extractor.extract_entities(job_description)
        jd_skills = jd_entities["skills"]

        jd_sbert = preprocessor.preprocess(job_description, mode="clean_only")
        jd_tfidf = preprocessor.preprocess(job_description, mode="full")
        job_embedding = matcher.embed(jd_sbert)

        resume_embeddings = []
        resume_names = []
        resume_skills_list = []
        resume_tfidf_texts = []
        resume_section_locations = []

        with st.spinner("⏳ Analysing resumes..."):
            for file in uploaded_files:
                raw_text = extract_pdf_text(file)
                if not raw_text.strip():
                    st.warning(f"⚠️ {file.name} has no readable text — skipped.")
                    continue

                sbert_text = preprocessor.preprocess(raw_text, mode="clean_only")
                embedding = matcher.embed(sbert_text)
                tfidf_text = preprocessor.preprocess(raw_text, mode="full")
                entities = skill_extractor.extract_entities(raw_text)
                skills = entities["skills"]
                sections = parse_sections(raw_text)
                skill_locations = locate_skills_in_sections(
                    sections, skills, skill_extractor
                )

                resume_embeddings.append(embedding)
                resume_names.append(file.name)
                resume_skills_list.append(skills)
                resume_tfidf_texts.append(tfidf_text)
                resume_section_locations.append(skill_locations)

        if not resume_embeddings:
            st.error("No valid resumes could be processed.")
        else:
            jd_vec, resume_vecs = tfidf_matcher.fit_and_embed_all(
                jd_tfidf, resume_tfidf_texts
            )
            tfidf_scores = [
                float(cosine_similarity([jd_vec], [v])[0][0])
                for v in resume_vecs
            ]

            ranking = rank_resumes(
                job_embedding=job_embedding,
                resume_embeddings=resume_embeddings,
                resume_names=resume_names,
                job_skills=jd_skills,
                resume_skills_list=resume_skills_list,
                tfidf_scores=tfidf_scores,
            )

            skill_locations_per_resume = dict(
                zip(resume_names, resume_section_locations)
            )

            # ── Results ───────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 Results")

            total_jd_skills = len(jd_skills) if jd_skills else 1

            if jd_skills:
                st.markdown(
                    f"**{len(jd_skills)} skills detected in JD:** "
                    + " ".join(f"`{s}`" for s in jd_skills)
                )
            else:
                st.warning(
                    "No specific skills detected in the JD. "
                    "Add explicit skill names like Python, Docker, AWS."
                )

            st.markdown(f"**{len(resume_embeddings)} resume(s) analysed.**")
            st.divider()

            # ── Resume cards ──────────────────────────────────────────────────
            for i, row in ranking.iterrows():
                score = float(row["Final Score"])
                pct = int(score * 100)
                label, _ = match_label(score)
                matched = row["Matched Skills"]
                missing = row["Missing Skills"]
                locations = skill_locations_per_resume.get(row["Resume"], {})
                skill_coverage = int((len(matched) / total_jd_skills) * 100)

                # All four features computed per resume
                interpretation = match_interpretation(score)       # Feature 2
                suggestions = improvement_suggestions(missing)     # Feature 1
                tips = section_tips(locations, matched)            # Feature 3
                report_text = generate_report(                     # Feature 4
                    resume_name=row["Resume"],
                    jd_text=job_description,
                    jd_skills=jd_skills,
                    score=score,
                    matched=matched,
                    missing=missing,
                    locations=locations,
                    suggestions=suggestions,
                    tips=tips
                )

                with st.container(border=True):

                    # ── Card header ───────────────────────────────────────────
                    h1, h2, h3 = st.columns([3, 1, 1])
                    with h1:
                        st.markdown(f"### #{i} {row['Resume']}")
                        st.markdown(f"{label}")
                        st.caption(interpretation)
                    with h2:
                        st.metric("Match Score", f"{pct}%")
                    with h3:
                        st.metric(
                            "Skill Coverage", f"{skill_coverage}%",
                            help="% of JD-required skills found in this resume"
                        )

                    # ── Score breakdown ───────────────────────────────────────
                    with st.expander("📐 Score Breakdown"):
                        s1, s2, s3 = st.columns(3)
                        s1.metric(
                            "SBERT Score",
                            f"{float(row['SBERT Score']):.3f}",
                            help="Semantic similarity from Sentence-BERT"
                        )
                        s2.metric(
                            "TF-IDF Score",
                            f"{float(row['TF-IDF Score']):.3f}",
                            help="Keyword overlap from TF-IDF"
                        )
                        s3.metric(
                            "Final Score",
                            f"{float(row['Final Score']):.3f}",
                            help="Hybrid: 80% SBERT + 20% TF-IDF"
                        )

                    # ── Skill match details ───────────────────────────────────
                    with st.expander(
                        "🔍 Skill Match Details",
                        expanded=(i == 1)
                    ):
                        sk1, sk2 = st.columns(2)
                        with sk1:
                            st.markdown(
                                f"**✅ Matched ({len(matched)}/{total_jd_skills})**"
                            )
                            if matched:
                                for skill in matched:
                                    section = locations.get(skill, "")
                                    label_text = (
                                        f"{skill}  `{section}`"
                                        if section else skill
                                    )
                                    st.success(label_text)
                            else:
                                st.info("No JD skills matched.")

                        with sk2:
                            st.markdown(
                                f"**❌ Missing ({len(missing)}/{total_jd_skills})**"
                            )
                            if missing:
                                for skill in missing:
                                    st.error(skill)
                            else:
                                st.success("🎉 All required JD skills present!")

                    # ── Feature 3: Section tips ───────────────────────────────
                    with st.expander("💡 Resume Section Tips"):
                        for tip in tips:
                            st.markdown(tip)

                    # ── Feature 1: Improvement suggestions ───────────────────
                    with st.expander("🚀 How to Improve This Resume"):
                        if suggestions:
                            st.markdown(
                                "Based on the missing skills, here are specific "
                                "steps to strengthen this resume:"
                            )
                            for suggestion in suggestions:
                                st.markdown(f"• {suggestion}")
                        else:
                            st.success(
                                "No missing skills — nothing to improve for this JD!"
                            )

                    # ── Feature 4: Download report ────────────────────────────
                    st.download_button(
                        label="📥 Download Full Report",
                        data=report_text,
                        file_name=f"report_{row['Resume'].replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )