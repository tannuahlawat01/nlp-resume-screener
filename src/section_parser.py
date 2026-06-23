import re

# Section header keywords mapped to canonical section names.
# Order matters — checked top to bottom, first match wins.
SECTION_PATTERNS = [
    (r"\b(work experience|experience|employment|professional experience|career)\b", "Experience"),
    (r"\b(education|academic background|qualifications)\b", "Education"),
    (r"\b(skills|technical skills|core competencies|technologies|tech stack)\b", "Skills"),
    (r"\b(projects|personal projects|academic projects|key projects)\b", "Projects"),
    (r"\b(internship|internships|training)\b", "Internship"),
    (r"\b(certifications|certificates|courses|achievements|awards)\b", "Certifications"),
    (r"\b(summary|objective|profile|about me)\b", "Summary"),
    (r"\b(publications|research|papers)\b", "Research"),
]


def parse_sections(text: str) -> dict:
    """
    Split resume text into sections based on header detection.

    Strategy:
      - Scan line by line for lines that look like section headers
        (short, mostly uppercase or matching known keywords).
      - Assign subsequent lines to that section until the next header.

    Args:
        text: raw resume text (not preprocessed)

    Returns:
        dict mapping section name -> section text
        e.g. {"Experience": "...", "Skills": "...", "Education": "..."}
        Always includes an "Other" key for unclassified content.
    """
    lines = text.splitlines()
    sections = {}
    current_section = "Other"
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue

        detected = _detect_header(stripped)

        if detected:
            # Save previous section
            if buffer:
                sections[current_section] = sections.get(current_section, "") + "\n".join(buffer)
            current_section = detected
            buffer = []
        else:
            buffer.append(stripped)

    # Save last section
    if buffer:
        sections[current_section] = sections.get(current_section, "") + "\n".join(buffer)

    return sections


def _detect_header(line: str) -> str | None:
    """
    Detect if a line is a section header.
    Returns canonical section name or None.

    A line is a header if:
      - It's short (under 40 chars)
      - AND it matches a known section keyword pattern
    """
    if len(line) > 40:
        return None

    line_lower = line.lower()
    for pattern, section_name in SECTION_PATTERNS:
        if re.search(pattern, line_lower):
            return section_name

    return None


def locate_skills_in_sections(
    sections: dict,
    skills: list,
    skill_extractor
) -> dict:
    """
    For each skill, find which resume section it appears in.

    Args:
        sections:        output of parse_sections()
        skills:          list of canonical skill names (from NER extractor)
        skill_extractor: SkillExtractor instance (to re-run on each section)

    Returns:
        dict mapping skill_name -> section_name
        e.g. {"Python": "Experience", "Docker": "Skills", "AWS": "Projects"}
    """
    skill_to_section = {}

    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue
        extracted = skill_extractor.extract_entities(section_text)
        for skill in extracted["skills"]:
            # First occurrence wins — most specific section
            if skill not in skill_to_section:
                skill_to_section[skill] = section_name

    return skill_to_section