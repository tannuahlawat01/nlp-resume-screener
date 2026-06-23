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
from src.ats_scorer import compute_ats_score, ats_grade


@st.cache_resource
def load_models():
    return SemanticMatcher(), TextPreprocessor(), SkillExtractor(), TFIDFMatcher()


matcher, preprocessor, skill_extractor, tfidf_matcher = load_models()


def match_interpretation(score: float) -> str:
    if score >= 0.55:
        return (
            "✅ Strong semantic match. Candidate covers most required "
            "skills and context. Recommended for shortlisting."
        )
    elif score >= 0.35:
        return (
            "⚠️ Partial match. Relevant experience present but key "
            "skills are missing. Review breakdown before deciding."
        )
    else:
        return (
            "❌ Weak match. Resume does not align well with this JD. "
            "Significant skill gaps detected."
        )


def improvement_suggestions(missing: list) -> list:
    suggestions = []
    for skill in missing:
        sl = skill.lower()
        if any(x in sl for x in ["docker", "kubernetes", "k8s"]):
            suggestions.append(
                f"**{skill}** — Add a project where you containerised an app. "
                f"Even a personal project using Docker Compose counts."
            )
        elif any(x in sl for x in ["aws", "gcp", "azure", "cloud"]):
            suggestions.append(
                f"**{skill}** — Get a free-tier account and deploy a project. "
                f"Mention the specific service used (EC2, S3, Lambda, etc.)."
            )
        elif any(x in sl for x in ["pytorch", "tensorflow", "keras"]):
            suggestions.append(
                f"**{skill}** — Build a small ML project and mention the "
                f"framework explicitly in your resume."
            )
        elif any(x in sl for x in ["sql", "postgresql", "mysql", "mongodb"]):
            suggestions.append(
                f"**{skill}** — If you've used any database in a project, "
                f"name it explicitly. Recruiters scan for database keywords."
            )
        elif any(x in sl for x in ["react", "node", "javascript", "typescript"]):
            suggestions.append(
                f"**{skill}** — Add to Skills section and mention a project "
                f"where you used it, even briefly."
            )
        elif any(x in sl for x in ["ci/cd", "github actions", "jenkins"]):
            suggestions.append(
                f"**{skill}** — Set up a basic GitHub Actions workflow on "
                f"one of your repos. 30 minutes of work, strong keyword signal."
            )
        elif any(x in sl for x in ["machine learning", "deep learning", "nlp"]):
            suggestions.append(
                f"**{skill}** — Describe your ML projects with specifics: "
                f"model type, dataset, metric. Vague descriptions don't register."
            )
        else:
            suggestions.append(
                f"**{skill}** — If you have any experience (coursework, "
                f"projects, self-study), add it explicitly to Skills or Projects."
            )
    return suggestions


def section_tips(locations: dict, matched: list) -> list:
    tips = []
    only_in_skills = [
        s for s in matched if locations.get(s, "") == "Skills"
    ]
    in_experience = [
        s for s in matched
        if locations.get(s, "") in ["Experience", "Internship", "Projects"]
    ]

    if only_in_skills:
        skill_list = ", ".join(only_in_skills[:4])
        tips.append(
            f"⚠️ **{skill_list}** appear only in your Skills section. "
            f"Mentioning them with context in Experience or Projects "
            f"significantly strengthens your resume — recruiters trust "
            f"demonstrated usage over listed skills."
        )
    if len(in_experience) == 0 and matched:
        tips.append(
            "⚠️ None of your matched skills appear in Experience. "
            "Try weaving key technologies into bullet points "
            "(e.g. 'Built REST API using FastAPI and PostgreSQL')."
        )
    if not tips:
        tips.append(
            "✅ Skills are well-distributed across sections. "
            "Good structure for both ATS and human reviewers."
        )
    return tips


def generate_report(
    resume_name, jd_skills, ats, score,
    matched, missing, locations, suggestions, tips
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    grade, label, _ = ats_grade(ats["total"])
    lines = [
        "=" * 60,
        "NLP RESUME SCREENER — ATS ANALYSIS REPORT",
        f"Generated: {now}",
        "=" * 60,
        "",
        f"Resume:       {resume_name}",
        f"ATS Score:    {ats['total']}/100  Grade: {grade} — {label}",
        f"Match Score:  {int(score * 100)}%",
        "",
        "SCORE BREAKDOWN",
        "-" * 40,
        f"  Skill Match:           {ats['skill_match']}/40",
        f"  Semantic Relevance:    {ats['semantic_relevance']}/25",
        f"  Experience Relevance:  {ats['experience_relevance']}/20",
        f"  Resume Structure:      {ats['resume_structure']}/10",
        f"  Keyword Coverage:      {ats['keyword_coverage']}/5",
        "",
        "JD SKILLS DETECTED",
        "-" * 40,
        ", ".join(jd_skills) if jd_skills else "None",
        "",
        f"MATCHED SKILLS ({len(matched)}/{len(jd_skills) if jd_skills else 0})",
        "-" * 40,
    ]
    for skill in matched:
        section = locations.get(skill, "Unknown")
        lines.append(f"  ✅ {skill:25s} [Found in: {section}]")

    lines += [
        "",
        f"MISSING SKILLS ({len(missing)}/{len(jd_skills) if jd_skills else 0})",
        "-" * 40,
    ]
    for skill in missing:
        lines.append(f"  ❌ {skill}")

    lines += ["", "IMPROVEMENT SUGGESTIONS", "-" * 40]
    for i, s in enumerate(suggestions, 1):
        lines.append(f"{i}. {s.replace('**', '')}")

    lines += ["", "SECTION TIPS", "-" * 40]
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
        "Helps **recruiters and job seekers** analyse how well a resume "
        "matches a job description using NLP.\n\n"
        "Combines **Sentence-BERT** (semantic understanding) with "
        "**TF-IDF** (keyword matching) into a multi-factor ATS score."
    )

    st.divider()
    st.markdown("## ⚙️ How It Works")
    st.markdown(
        "1. **Upload** resume PDF(s)\n"
        "2. **Paste** the job description\n"
        "3. **Click** Rank Resumes\n\n"
        "The app extracts skills, parses resume sections, computes "
        "semantic similarity, and produces an explainable ATS score."
    )

    st.divider()
    st.markdown("## 📊 ATS Score Breakdown")
    st.markdown(
        "| Factor | Weight |\n"
        "|--------|--------|\n"
        "| Skill Match | 40 pts |\n"
        "| Semantic Relevance | 25 pts |\n"
        "| Experience Relevance | 20 pts |\n"
        "| Resume Structure | 10 pts |\n"
        "| Keyword Coverage | 5 pts |"
    )

    st.divider()
    st.markdown("## 🏷️ Grade Guide")
    st.markdown(
        "🟢 **A** — 80+ Excellent Match\n\n"
        "🔵 **B** — 65–79 Good Match\n\n"
        "🟡 **C** — 50–64 Moderate Match\n\n"
        "🔴 **D** — 35–49 Weak Match\n\n"
        "🔴 **F** — Below 35 Poor Match"
    )

    st.divider()
    st.markdown("## 📈 Model Benchmark")
    eval_path = os.path.join("evaluation", "results", "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        avg = next(r for r in eval_data if r["JD"] == "AVERAGE")
        c1, c2 = st.columns(2)
        c1.metric(
            "SBERT P@1",
            f"{float(avg['SBERT P@1']):.0%}",
            f"+{(float(avg['SBERT P@1']) - float(avg['TFIDF P@1'])):.0%} vs TF-IDF"
        )
        c2.metric("TF-IDF P@1", f"{float(avg['TFIDF P@1']):.0%}")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📄 NLP Resume Screener")
st.markdown(
    "#### Rank candidates and improve resumes using semantic AI — "
    "not just keyword matching."
)
st.info(
    "💡 **Recruiters:** Upload multiple resumes to rank and score candidates. "
    "**Job seekers:** Upload your resume to get an ATS score, skill gap "
    "analysis, and specific improvement tips.",
    icon="💡"
)
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📁 Upload Resume(s)")
    st.caption("PDF only. Upload one (job seeker) or many (recruiter).")
    uploaded_files = st.file_uploader(
        "Resumes", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) ready.")

with col_right:
    st.markdown("### 📋 Job Description")
    st.caption("Paste the full JD including required and nice-to-have skills.")
    job_description = st.text_area(
        "JD", height=220,
        placeholder="e.g. Looking for a Python backend engineer with FastAPI, PostgreSQL, Docker and AWS...",
        label_visibility="collapsed"
    )

st.divider()
run_col, _ = st.columns([1, 3])
with run_col:
    run_button = st.button(
        "🔍 Analyse Resumes", type="primary", use_container_width=True
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
        resume_detected_sections = []

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
                resume_detected_sections.append(list(sections.keys()))

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
            detected_sections_per_resume = dict(
                zip(resume_names, resume_detected_sections)
            )
            tfidf_scores_map = dict(zip(resume_names, tfidf_scores))

            # ── Results ───────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 ATS Analysis Results")

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

            # ── Summary leaderboard ───────────────────────────────────────────
            st.markdown("### 🏆 Candidate Leaderboard")
            leaderboard_rows = []
            ats_results_map = {}

            for i, row in ranking.iterrows():
                name = row["Resume"]
                sbert_score = float(row["SBERT Score"])
                tfidf_score = tfidf_scores_map.get(name, 0.0)
                locations = skill_locations_per_resume.get(name, {})
                sections = detected_sections_per_resume.get(name, [])

                ats = compute_ats_score(
                    jd_skills=jd_skills,
                    resume_skills=resume_skills_list[resume_names.index(name)],
                    sbert_score=sbert_score,
                    tfidf_score=tfidf_score,
                    skill_locations=locations,
                    detected_sections=sections,
                )
                ats_results_map[name] = ats
                grade, label, _ = ats_grade(ats["total"])

                leaderboard_rows.append({
                    "Rank": f"#{i}",
                    "Resume": name,
                    "ATS Score": f"{ats['total']}/100",
                    "Grade": f"{grade} — {label}",
                    "Skill Match": f"{ats['breakdown']['skill_match_pct']}%",
                    "Matched": len(ats['breakdown']['matched_skills']),
                    "Missing": len(ats['breakdown']['missing_skills']),
                })

            lb_df = pd.DataFrame(leaderboard_rows)
            st.dataframe(lb_df, use_container_width=True, hide_index=True)
            st.divider()

            # ── Detailed cards ────────────────────────────────────────────────
            st.markdown("### 📋 Detailed Analysis")

            for i, row in ranking.iterrows():
                name = row["Resume"]
                score = float(row["Final Score"])
                sbert_score = float(row["SBERT Score"])
                tfidf_score = tfidf_scores_map.get(name, 0.0)
                locations = skill_locations_per_resume.get(name, {})
                sections = detected_sections_per_resume.get(name, [])
                ats = ats_results_map[name]
                grade, grade_label, grade_color = ats_grade(ats["total"])
                matched = ats["breakdown"]["matched_skills"]
                missing = ats["breakdown"]["missing_skills"]

                interpretation = match_interpretation(score)
                suggestions = improvement_suggestions(missing)
                tips = section_tips(locations, matched)
                report_text = generate_report(
                    name, jd_skills, ats, score,
                    matched, missing, locations, suggestions, tips
                )

                with st.container(border=True):

                    # Card header
                    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
                    with h1:
                        st.markdown(f"### #{i} {name}")
                        st.caption(interpretation)
                    with h2:
                        st.metric("ATS Score", f"{ats['total']}/100")
                    with h3:
                        st.metric("Grade", f"{grade} — {grade_label}")
                    with h4:
                        st.metric(
                            "Skill Coverage",
                            f"{ats['breakdown']['skill_match_pct']}%"
                        )

                    # ATS score breakdown bar
                    st.markdown("**Score Breakdown:**")
                    b1, b2, b3, b4, b5 = st.columns(5)
                    b1.metric(
                        "Skill Match",
                        f"{ats['skill_match']}/40",
                        help="JD skills found in resume"
                    )
                    b2.metric(
                        "Semantic",
                        f"{ats['semantic_relevance']}/25",
                        help="SBERT semantic similarity"
                    )
                    b3.metric(
                        "Experience",
                        f"{ats['experience_relevance']}/20",
                        help="Skills found in Experience/Projects sections"
                    )
                    b4.metric(
                        "Structure",
                        f"{ats['resume_structure']}/10",
                        help="Key sections present in resume"
                    )
                    b5.metric(
                        "Keywords",
                        f"{ats['keyword_coverage']}/5",
                        help="TF-IDF keyword overlap"
                    )

                    # Skill match details
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

                    # Section tips
                    with st.expander("💡 Resume Section Tips"):
                        for tip in tips:
                            st.markdown(tip)

                    # Improvement suggestions
                    with st.expander("🚀 How to Improve This Resume"):
                        if suggestions:
                            st.markdown(
                                "Based on missing skills, here are specific steps:"
                            )
                            for s in suggestions:
                                st.markdown(f"• {s}")
                        else:
                            st.success("No missing skills — nothing to improve!")

                    # Download report
                    st.download_button(
                        label="📥 Download Full ATS Report",
                        data=report_text,
                        file_name=f"ats_report_{name.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )