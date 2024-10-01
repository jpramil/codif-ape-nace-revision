SYS_PROMPT = """You are an expert in the NACE. Your goal is:

1. Analyze the job title and job description provided by the user.
2. From the list of occupational categories provided, identify the most appropriate ISCO code (4 digits) based on the job description. If the job description is not clear, use the job title to classify the job.
3. Return the 4-digit code in JSON format as specified by the user. If the job cannot be classified within the given categories, return `null` in the JSON.
"""

CLASSIF_PROMPT = """\
- Activité de l'entreprise :
{activity}

- Liste des codes NACE potentiels et leurs notes explicatives :
{proposed_codes}

{format_instructions}
"""


def format_codes(codes: list):
    return "\n\n".join(
        [
            f"{nace2025.code}: {nace2025.label}\n{extract_info(nace2025, paragraphs=["include", "not_include", "notes"])}"
            for nace2025 in codes
        ]
    )
    

def extract_info(nace2025, paragraphs=["include", "not_include", "notes"]):
    info = [getattr(nace2025, paragraph) for paragraph in paragraphs if getattr(nace2025, paragraph) is not None]
    return "\n\n".join(info) if info else ""


def generate_prompt(row, mapping, parser):
    nace08 = row.apet_finale
    activity = row.libelle_activite

    proposed_codes = next((m.naf2025 for m in mapping if m.code == nace08))
    prompt = CLASSIF_PROMPT.format(
            **{
                "activity": activity,
                "proposed_codes": format_codes(proposed_codes),
                "format_instructions": parser.get_format_instructions(),
            }
        )
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]