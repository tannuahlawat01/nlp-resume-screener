# NLP Resume Screener — Semantic Candidate Matching Engine

An intelligent NLP-powered resume screening system that extracts text from resumes, preprocesses content, identifies technical skills, generates semantic embeddings, and ranks candidates against a job description using cosine similarity.

This project demonstrates real-world NLP pipeline engineering, clean architecture principles, and ML-driven semantic matching — designed to simulate a recruiter workflow.

---

## 🚀 Features

* 📄 Resume text extraction from PDF and DOCX files
* 🧹 NLP preprocessing (cleaning, normalization, lemmatization)
* 🧠 Skill & role extraction using Named Entity Recognition
* 🔍 Semantic embedding with Sentence-BERT
* 📊 Cosine similarity ranking engine
* 🌐 Streamlit recruiter dashboard
* 🧱 Clean modular architecture

---

## 🧠 Architecture Overview

The system follows clean architecture principles to ensure modularity, maintainability, and scalability.

```
UI Layer (Streamlit Dashboard)
        ↓
Ranking Engine (Business Logic)
        ↓
Semantic Embeddings
        ↓
NLP Pipeline (Cleaning + NER)
        ↓
Resume Extraction Layer
```

Each layer is independent and testable. Business logic is separated from UI, allowing easy upgrades or replacement of components.

---

## 📦 Installation

### 1. Clone the repository

```
git clone <your-repository-url>
cd nlp-resume-screener
```

### 2. Create a virtual environment (recommended)

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Download spaCy language model

```
python -m spacy download en_core_web_sm
```

---

## ▶ Usage

### Run CLI pipeline demo

Processes resumes and demonstrates extraction → preprocessing → NER → embeddings → ranking.

```
python main.py
```

---

### Launch recruiter dashboard

Upload resumes, paste a job description, and view ranked candidates.

```
streamlit run app.py
```

---

## 📊 How Matching Works

1. Resume text and job description are converted into semantic embeddings
2. Cosine similarity measures semantic closeness
3. Scores are normalized into match percentages
4. Candidates are ranked and displayed

This allows meaning-based comparison instead of simple keyword matching.

---

## 📁 Project Structure

```
nlp-resume-screener/
│
├── app.py                  # Streamlit dashboard (UI layer)
├── main.py                 # CLI pipeline demo
│
├── data/
│   └── resumes/            # Input resume files
│
├── outputs/
│   └── extracted_texts/    # Saved processed text
│
├── src/
│   ├── extractor.py        # Resume file parsing
│   ├── preprocessing.py    # NLP cleaning pipeline
│   ├── ner_extractor.py    # Skill/entity detection
│   ├── embedding.py        # Semantic vectorization
│   ├── ranking_engine.py   # Matching & scoring logic
│   └── __init__.py
│
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies Used

* Python
* spaCy NLP
* Sentence-BERT embeddings
* Scikit-learn similarity engine
* Pandas data processing
* Streamlit dashboard
* pdfplumber & python-docx extraction

---

## 🎯 Learning Outcomes

* NLP pipeline design
* Semantic search & embeddings
* Clean architecture practices
* Modular ML system design
* Practical recruiter workflow simulation

---

## 🔮 Future Improvements

* Skill weighting & scoring customization
* Multi-job comparison engine
* Resume summarization
* REST API integration
* Cloud deployment
* Recruiter analytics dashboard

---

## 👩‍💻 Author

Built as an end-to-end NLP project demonstrating semantic resume matching and recruiter workflow automation.

---

## ⭐ Support

If you found this project useful:

* Star the repository
* Share feedback
* Connect for collaboration

---
