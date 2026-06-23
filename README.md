# 📄 NLP Resume Screener

An intelligent resume screening system that ranks candidates against a job description using **Sentence-BERT semantic matching** — not just keyword overlap. Produces a multi-factor **ATS Score (0–100)** with explainable skill breakdowns and resume improvement tips.

🔗 **[Live Demo](https://nlp-resume-screener-ifk6k3gdaaxp6rvc3neipo.streamlit.app)**  
📁 **[GitHub](https://github.com/tannuahlawat01/nlp-resume-screener)**

---

## 📊 Evaluation Results

| Metric | SBERT | TF-IDF | Improvement |
|--------|-------|--------|-------------|
| Precision@1 | **100%** | 66.7% | +33.3% |
| MRR | **1.000** | 0.833 | +0.167 |
| NDCG@3 | **0.994** | 0.910 | +0.084 |

Evaluated on 3 labeled JD-resume benchmark pairs (12 resumes total).  
SBERT correctly ranked the most relevant resume #1 in all 3 cases.  
TF-IDF failed on JD_001 — matched a Java engineer to a Python backend role due to surface keyword overlap (PostgreSQL, REST API appeared in both).

---

## 🧠 How It Works

### Multi-Factor ATS Score (100 points)

| Factor | Weight | Signal |
|--------|--------|--------|
| Skill Match | 40 pts | JD skills found in resume |
| Semantic Relevance | 25 pts | SBERT cosine similarity (rescaled) |
| Experience Relevance | 20 pts | Skills found in Experience/Projects sections |
| Resume Structure | 10 pts | Key sections present (Experience, Skills, Education, Projects) |
| Keyword Coverage | 5 pts | TF-IDF overlap score (rescaled) |

**Why these weights?** Skill match dominates (40%) as the most direct signal. Experience relevance (20%) uses the section parser to distinguish demonstrated skill usage from listed keywords — a skill in Experience is stronger evidence than the same skill in a Skills section. This mirrors how real ATS systems weight candidates.

### Preprocessing Strategy
Two modes — not one:
- **`clean_only`** for SBERT: preserves natural language structure. SBERT is a transformer trained on full sentences — lemmatization destroys the contextual signals it relies on.
- **`full`** for TF-IDF: lemmatize + remove stopwords. TF-IDF is bag-of-words, so lemmatization reduces sparsity.

### Skill Extraction
- 174-pattern spaCy EntityRuler taxonomy across 7 categories (languages, frameworks, ML/AI, cloud, DevOps, databases, tools)
- rapidfuzz fallback for variant normalization: `Postgres → PostgreSQL`, `ReactJS → React`, `K8s → Kubernetes`
- Two-layer pipeline: exact EntityRuler match first, fuzzy match for anything missed

---

## ✨ Features

- **Candidate Leaderboard** — ranked table with ATS Score, Grade, Skill Match %
- **Explainable Skill Breakdown** — matched vs missing skills, with the resume section each skill was found in (`Python \`Experience\``)
- **Resume Section Parser** — detects Experience, Education, Skills, Projects, Internship, Certifications, Summary, Research sections
- **Improvement Suggestions** — skill-specific actionable advice for each missing JD skill
- **Section Tips** — flags skills only in Skills section, not demonstrated in Experience
- **Download ATS Report** — full plain-text analysis report per candidate
- **Model Comparison Chart** — live benchmark of SBERT vs TF-IDF in sidebar

---

## 🏗 Architecture

```
app.py  (Streamlit UI)
    │
    ├── src/ats_scorer.py        # Multi-factor ATS scoring (5 signals)
    ├── src/ranking_engine.py    # Hybrid ranking (SBERT + TF-IDF)
    ├── src/embedding.py         # Sentence-BERT embeddings
    ├── src/tfidf_matcher.py     # TF-IDF baseline matcher
    ├── src/ner_extractor.py     # Skill extraction + fuzzy normalization
    ├── src/section_parser.py    # Resume section detection
    ├── src/preprocessing.py     # Dual-mode preprocessor
    ├── src/extractor.py         # PDF text extraction
    └── src/skill_patterns.json  # 174-pattern skill taxonomy
    │
    └── evaluation/
        ├── evaluate.py          # SBERT vs TF-IDF benchmark harness
        ├── labeled_dataset.py   # 3 JDs × 4 resumes labeled dataset
        └── results/
            └── eval_results.json
```

---

## 📦 Installation

```bash
git clone https://github.com/tannuahlawat01/nlp-resume-screener.git
cd nlp-resume-screener
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## ▶ Usage

### Run the app locally
```bash
streamlit run app.py
```

### Run evaluation benchmark
```bash
python -m evaluation.evaluate
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Semantic Matching | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Skill Extraction | spaCy EntityRuler + rapidfuzz |
| Baseline Model | TF-IDF + cosine similarity |
| NLP Preprocessing | spaCy, regex |
| PDF Extraction | pdfplumber |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📉 Known Limitations

**1. Semantic synonyms across domains**  
"ML Engineer" and "AI Developer" are semantically similar but the skill taxonomy treats them as different roles. A resume using non-standard terminology may score lower despite being relevant.

**2. Scanned PDF resumes**  
pdfplumber extracts text from digital PDFs only. Scanned/image-based PDFs produce empty text and are skipped. OCR support would be needed for these.

**3. Small evaluation benchmark**  
The labeled dataset has 3 JDs and 12 resumes — sufficient to demonstrate SBERT's advantage over TF-IDF, but not large enough for statistically robust conclusions. A production system would need 50+ labeled pairs.

**4. ATS weight calibration**  
The 5-factor weights (40/25/20/10/5) are based on documented reasoning but not empirically optimized. A larger labeled dataset would allow weight tuning via grid search or learned ranking.

---

## 🔮 Planned Improvements

- Section-weighted scoring (skill in Experience > skill in Skills section)
- Weight optimization via ablation study on expanded labeled dataset
- Support for DOCX resume uploads
- REST API wrapper for integration with HR tools

---

## 👩‍💻 Author

**Tannu Ahlawat**  
B.Tech AI & ML, IGDTUW Delhi (2028)  
GitHub: [@tannuahlawat01](https://github.com/tannuahlawat01)
