import re
import spacy
class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text):
        text = re.sub(r"http\S+|www\S+", " ", text)      
        text = re.sub(r"\S+@\S+", " ", text)             
        text = re.sub(r"[^a-zA-Z\s]", " ", text)         
        text = re.sub(r"\s+", " ", text).strip()         
        return text.lower()

    def preprocess(self, text):
        cleaned = self.clean_text(text)
        doc = self.nlp(cleaned)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        return " ".join(tokens)
