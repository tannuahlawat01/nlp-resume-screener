from src.section_parser import parse_sections, locate_skills_in_sections
from src.ner_extractor import SkillExtractor

extractor = SkillExtractor()

resume = """
Summary
Python developer with 2 years of experience in backend systems.

Experience
Built REST APIs using FastAPI and PostgreSQL at Sansoftech.
Deployed services on AWS EC2 with Docker containers.

Skills
Python, FastAPI, PostgreSQL, Docker, AWS, Git

Education
B.Tech in AI & ML, IGDTUW Delhi, 2028
"""

sections = parse_sections(resume)
print("Sections found:", list(sections.keys()))

skill_locations = locate_skills_in_sections(sections, [], extractor)
print("\nSkill locations:")
for skill, section in sorted(skill_locations.items()):
    print(f"  {skill:20s} → {section}")