import spacy
import json
import os
class SkillExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        base_dir = os.path.dirname(__file__)
        pattern_path = os.path.join(base_dir, "skill_patterns.json")

        with open(pattern_path, "r", encoding="utf-8") as f:
            patterns = json.load(f)
        ruler.add_patterns(patterns)

    def extract_entities(self, text):
        doc = self.nlp(text)
        skills = []
        roles = []
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                skills.append(ent.text)
            elif ent.label_ == "ROLE":
                roles.append(ent.text)

        return {
            "skills": list(set(skills)),
            "roles": list(set(roles))
        }

