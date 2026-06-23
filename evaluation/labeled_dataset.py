"""
Labeled evaluation dataset.
Each JD is paired with 4 resumes ranked by human relevance (1 = most relevant).
"""

LABELED_DATA = [
    {
        "jd_id": "JD_001",
        "job_description": """
            We are looking for a Python Backend Engineer with experience in FastAPI or Django,
            PostgreSQL, Docker, and REST API design. Knowledge of AWS or GCP is a plus.
            The candidate should have strong Git skills and experience with CI/CD pipelines.
            Experience with Redis for caching and system design principles required.
        """,
        "resumes": [
            {
                "id": "R1",
                "text": """
                    Software Engineer with 2 years of experience in Python and FastAPI.
                    Built and deployed REST APIs using PostgreSQL and Redis on AWS EC2.
                    Strong Docker and CI/CD experience using GitHub Actions.
                    Familiar with system design and microservices architecture.
                """,
                "relevance_rank": 1
            },
            {
                "id": "R2",
                "text": """
                    Full Stack Developer skilled in Python, Django, and MySQL.
                    Experience with Git, Linux, and basic Docker usage.
                    Built web applications using React and Node.js frontend.
                    Some exposure to AWS S3 for file storage.
                """,
                "relevance_rank": 2
            },
            {
                "id": "R3",
                "text": """
                    Java Spring Boot engineer with PostgreSQL and Kubernetes experience.
                    Strong backend skills but primary language is Java, not Python.
                    CI/CD using Jenkins and Terraform for infrastructure.
                    Knowledge of REST API design and microservices.
                """,
                "relevance_rank": 3
            },
            {
                "id": "R4",
                "text": """
                    Frontend Developer with expertise in React, TypeScript, and Tailwind CSS.
                    Experience with Figma and UI/UX design principles.
                    Basic JavaScript and HTML/CSS skills.
                    No backend or database experience.
                """,
                "relevance_rank": 4
            },
        ]
    },
    {
        "jd_id": "JD_002",
        "job_description": """
            Machine Learning Engineer role requiring PyTorch, Hugging Face Transformers,
            and experience fine-tuning LLMs or BERT-based models. Strong Python skills
            required. Familiarity with FAISS, LangChain, or RAG pipelines is a plus.
            Experience with MLflow for experiment tracking and deployment on cloud.
        """,
        "resumes": [
            {
                "id": "R5",
                "text": """
                    ML Engineer with deep expertise in PyTorch and Hugging Face Transformers.
                    Fine-tuned BERT and GPT models for NLP classification tasks.
                    Built RAG pipeline using FAISS and LangChain. Tracked experiments with MLflow.
                    Deployed models on AWS SageMaker and GCP Vertex AI.
                """,
                "relevance_rank": 1
            },
            {
                "id": "R6",
                "text": """
                    Data Scientist with strong Python and scikit-learn background.
                    Experience with XGBoost, LightGBM for tabular ML tasks.
                    Some exposure to TensorFlow for deep learning projects.
                    Used Pandas, NumPy, Matplotlib for data analysis.
                """,
                "relevance_rank": 2
            },
            {
                "id": "R7",
                "text": """
                    Software Engineer with Python and Flask experience.
                    Built REST APIs and worked with PostgreSQL databases.
                    No ML or deep learning experience listed.
                    Familiar with Docker, Git, and Linux environments.
                """,
                "relevance_rank": 3
            },
            {
                "id": "R8",
                "text": """
                    Android Developer skilled in Kotlin and Java.
                    Built mobile applications using Firebase and REST APIs.
                    Experience with Agile and Scrum methodologies.
                    No machine learning or Python background.
                """,
                "relevance_rank": 4
            },
        ]
    },
    {
        "jd_id": "JD_003",
        "job_description": """
            Full Stack Developer needed with React, Node.js, TypeScript, and PostgreSQL skills.
            Must have experience with REST APIs, authentication using JWT or OAuth, and deployment
            on Vercel or similar platforms. GraphQL knowledge is a bonus.
            Strong Git workflow and Agile experience required.
        """,
        "resumes": [
            {
                "id": "R9",
                "text": """
                    Full Stack Developer with 3 years in React, TypeScript, and Node.js.
                    Built production apps with PostgreSQL, JWT auth, and GraphQL APIs.
                    Deployed on Vercel and Railway. Strong Git and Agile workflow experience.
                """,
                "relevance_rank": 1
            },
            {
                "id": "R10",
                "text": """
                    Frontend Developer specializing in React and JavaScript.
                    Experience with REST API integration and basic Node.js.
                    Used Firebase for auth instead of JWT. Deployed on Netlify.
                    Familiar with Git and basic Agile processes.
                """,
                "relevance_rank": 2
            },
            {
                "id": "R11",
                "text": """
                    Backend Engineer with Django and Python REST API experience.
                    PostgreSQL and Redis user. Some Docker and Linux background.
                    No React or frontend experience. Prefers Python over JavaScript.
                """,
                "relevance_rank": 3
            },
            {
                "id": "R12",
                "text": """
                    Data Analyst with SQL and Excel skills.
                    Built dashboards using Tableau and Power BI.
                    No web development or programming beyond SQL queries.
                """,
                "relevance_rank": 4
            },
        ]
    },
]