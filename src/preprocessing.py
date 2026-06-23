import re
import spacy


class TextPreprocessor:
    """
    Two preprocessing modes:

    - clean_only (default, for SBERT):
        Remove noise (URLs, emails, special chars) but keep natural sentence
        structure. SBERT is a transformer trained on natural language — feeding
        it lemmatized, stopword-stripped text destroys the contextual signals
        it relies on. "I have experience in machine learning" scores much
        better than "experi machin learn".

    - full (for TF-IDF):
        Clean + lemmatize + remove stopwords. TF-IDF is bag-of-words so
        lemmatization reduces vocabulary sparsity and stopwords add noise
        without adding meaning.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text: str) -> str:
        """
        Remove noise common in resumes:
        - URLs and email addresses
        - Special characters (but keep +, #, . for C++, C#, Node.js)
        - Normalize whitespace
        """
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s\+\#\.]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: str, mode: str = "clean_only") -> str:
        """
        Preprocess text for downstream ML use.

        Args:
            text: raw input text (resume or job description)
            mode: preprocessing strategy —
                  'clean_only' → for SBERT (default)
                  'full'       → for TF-IDF

        Returns:
            Preprocessed string ready for embedding or vectorization.

        Raises:
            ValueError: if mode is not 'clean_only' or 'full'
        """
        cleaned = self.clean_text(text)

        if mode == "clean_only":
            # Preserve natural language for transformer models
            return cleaned.lower()

        elif mode == "full":
            # Lemmatize + remove stopwords for bag-of-words models
            doc = self.nlp(cleaned.lower())
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and len(token.text) > 1
            ]
            return " ".join(tokens)

        else:
            raise ValueError(
                f"Unknown mode: '{mode}'. Use 'clean_only' or 'full'."
            )