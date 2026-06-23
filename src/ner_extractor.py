import spacy
import json
import os
from rapidfuzz import process, fuzz

# Canonical display names: pattern id -> display name
SKILL_CANONICAL = {
    "python": "Python", "java": "Java", "javascript": "JavaScript",
    "typescript": "TypeScript", "cpp": "C++", "csharp": "C#",
    "golang": "Go", "rust": "Rust", "ruby": "Ruby", "php": "PHP",
    "swift": "Swift", "kotlin": "Kotlin", "scala": "Scala",
    "r_lang": "R", "matlab": "MATLAB", "bash": "Bash", "shell": "Shell",
    "powershell": "PowerShell", "html": "HTML", "css": "CSS",
    "sql": "SQL", "nosql": "NoSQL",
    "react": "React", "nextjs": "Next.js", "vue": "Vue.js",
    "angular": "Angular", "nodejs": "Node.js", "express": "Express",
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
    "springboot": "Spring Boot", "spring": "Spring", "laravel": "Laravel",
    "rails": "Ruby on Rails", "graphql": "GraphQL", "rest": "REST API",
    "grpc": "gRPC", "tailwind": "Tailwind CSS", "bootstrap": "Bootstrap",
    "redux": "Redux",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "keras": "Keras",
    "sklearn": "scikit-learn", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "catboost": "CatBoost", "huggingface": "Hugging Face",
    "transformers": "Transformers", "bert": "BERT", "gpt": "GPT",
    "llm": "LLM", "langchain": "LangChain", "llamaindex": "LlamaIndex",
    "rag": "RAG", "faiss": "FAISS", "opencv": "OpenCV", "nltk": "NLTK",
    "spacy": "spaCy", "pandas": "Pandas", "numpy": "NumPy",
    "matplotlib": "Matplotlib", "seaborn": "Seaborn", "plotly": "Plotly",
    "ml": "Machine Learning", "dl": "Deep Learning", "nlp": "NLP",
    "cv": "Computer Vision", "rl": "Reinforcement Learning",
    "mlflow": "MLflow", "wandb": "Weights & Biases", "sbert": "Sentence-BERT",
    "aws": "AWS", "gcp": "GCP", "azure": "Azure", "ec2": "EC2",
    "s3": "S3", "lambda": "AWS Lambda", "vercel": "Vercel",
    "heroku": "Heroku", "railway": "Railway", "netlify": "Netlify",
    "firebase": "Firebase", "supabase": "Supabase",
    "docker": "Docker", "kubernetes": "Kubernetes", "cicd": "CI/CD",
    "github_actions": "GitHub Actions", "jenkins": "Jenkins",
    "terraform": "Terraform", "ansible": "Ansible", "git": "Git",
    "github": "GitHub", "gitlab": "GitLab", "linux": "Linux",
    "unix": "Unix", "nginx": "Nginx", "apache": "Apache",
    "postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB",
    "redis": "Redis", "sqlite": "SQLite", "cassandra": "Cassandra",
    "elasticsearch": "Elasticsearch", "prisma": "Prisma",
    "sqlalchemy": "SQLAlchemy", "pinecone": "Pinecone",
    "chromadb": "ChromaDB", "weaviate": "Weaviate", "kafka": "Kafka",
    "spark": "Apache Spark",
    "streamlit": "Streamlit", "gradio": "Gradio", "jupyter": "Jupyter",
    "postman": "Postman", "vscode": "VS Code", "figma": "Figma",
    "jira": "Jira", "websockets": "WebSockets", "stripe": "Stripe",
    "clerk": "Clerk", "oauth": "OAuth", "jwt": "JWT",
    "microservices": "Microservices", "system_design": "System Design",
    "agile": "Agile", "scrum": "Scrum",
}

# Module-level cache — loaded once, reused across calls
_FUZZY_CHOICES = None
_PATTERN_DATA = None


def _load_patterns():
    """Load pattern data once and cache it."""
    global _FUZZY_CHOICES, _PATTERN_DATA
    if _FUZZY_CHOICES is not None:
        return
    base_dir = os.path.dirname(__file__)
    pattern_path = os.path.join(base_dir, "skill_patterns.json")
    with open(pattern_path, "r", encoding="utf-8") as f:
        _PATTERN_DATA = json.load(f)
    _FUZZY_CHOICES = [p["pattern"] for p in _PATTERN_DATA]


def fuzzy_extract_skills(text: str, threshold: int = 85) -> list:
    """
    Fuzzy-match tokens and bigrams in text against known skill patterns.
    Catches variants the entity ruler misses: 'Postgres' -> 'PostgreSQL',
    'ReactJS' -> 'React', 'K8s' -> 'Kubernetes'.

    Args:
        text: raw or lightly cleaned input text
        threshold: minimum match score (0-100). 85 balances recall vs false positives.

    Returns:
        Sorted list of canonical skill display names.
    """
    _load_patterns()
    found_ids = set()
    found_names = set()

    words = text.split()

    # Build unigram + bigram candidates
    candidates = words + [
        " ".join(words[i:i + 2]) for i in range(len(words) - 1)
    ]
    candidates = [c.strip() for c in candidates if len(c) >= 2]

    pattern_lookup = {p["pattern"].lower(): p for p in _PATTERN_DATA}

    for candidate in candidates:
        # Layer 1: exact case-insensitive match
        if candidate.lower() in pattern_lookup:
            p = pattern_lookup[candidate.lower()]
            pid = p.get("id", p["pattern"])
            if pid not in found_ids:
                found_ids.add(pid)
                found_names.add(SKILL_CANONICAL.get(pid, p["pattern"]))
            continue

        # Layer 2: fuzzy match for variants (min length 4 to avoid noise)
        if len(candidate) >= 4:
            result = process.extractOne(
                candidate,
                _FUZZY_CHOICES,
                scorer=fuzz.token_sort_ratio
            )
            if result and result[1] >= threshold:
                match = result[0]
                p = next(x for x in _PATTERN_DATA if x["pattern"] == match)
                pid = p.get("id", p["pattern"])
                if pid not in found_ids:
                    found_ids.add(pid)
                    found_names.add(SKILL_CANONICAL.get(pid, p["pattern"]))

    return sorted(found_names)


class SkillExtractor:
    """
    Two-layer skill extractor:
      1. spaCy EntityRuler: fast, exact pattern matching
      2. rapidfuzz fallback: catches spelling variants and abbreviations
    Results are merged and deduplicated using canonical IDs.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        base_dir = os.path.dirname(__file__)
        pattern_path = os.path.join(base_dir, "skill_patterns.json")
        with open(pattern_path, "r", encoding="utf-8") as f:
            patterns = json.load(f)
        ruler.add_patterns(patterns)

    def extract_entities(self, text: str) -> dict:
        """
        Extract skills and roles from text.

        Returns:
            {
                "skills": ["Python", "AWS", "PostgreSQL", ...],  # sorted, canonical
                "roles":  ["Software Engineer", ...]
            }
        """
        doc = self.nlp(text)
        ruler_skills = set()
        roles = set()

        for ent in doc.ents:
            if ent.label_ == "SKILL":
                pid = ent.ent_id_ or ent.text
                ruler_skills.add(SKILL_CANONICAL.get(pid, ent.text))
            elif ent.label_ == "ROLE":
                roles.add(ent.text)

        # Supplement with fuzzy matching to catch variants ruler missed
        fuzzy_skills = set(fuzzy_extract_skills(text))
        all_skills = sorted(ruler_skills | fuzzy_skills)

        return {
            "skills": all_skills,
            "roles": sorted(roles)
        }