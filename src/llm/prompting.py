from collections import namedtuple

PromptData = namedtuple("PromptData", ["id", "proposed_codes", "prompt"])

SYS_PROMPT = """You are an expert in the Statistical Classification of Economic Activities in the European Community (NACE). You are responsible for transitioning to a new classification. Your mission is to assign a NACE 2025 code to a company based on the description of its activity and a list of proposed codes (identified from its existing NACE 2008 code). Here are the instructions to follow:

1. Analyze the description of the company's main activity and the NACE 2008 code provided by the user in French.
2. Based on the available list of NACE 2025 codes, identify the most appropriate category that corresponds to the company's main activity.
3. Return the NACE 2025 code in JSON format as specified by the user. If the description of the company's activity is not sufficiently detailed to identify an appropriate NACE 2025 code, return `null` in the JSON.
4. Evaluate the consistency between the provided NACE 2008 code and the description of the company's activity. If the NACE 2008 code does not seem to match this description, return `False` in the `nace08_valid` field of the JSON. Note that if you are able to classify the description of the company's activity under a NACE 2025 code, the `nace08_valid` field should be `True`; otherwise, there is an inconsistency.
5. Respond only with the completed JSON; no additional information should be returned.
"""

CLASSIF_PROMPT = """\
- Main activity of the company:
{activity}

- Previous assigned NACE 2008 code:
{nace08}

- List of potential NACE codes and their explanatory notes:
{proposed_codes}

{format_instructions}
"""


# SYS_PROMPT = """Tu es un expert de la Nomenclature statistique des Activités économiques dans la Communauté Européenne (NACE). Tu es chargé de réaliser le changement de nomenclatureTa mission consiste à attribuer un code NACE 2025 à une entreprise, en t'appuyant sur le descriptif de son activité et à partir d'une liste de codes proposés (identifiée à partir de son code NACE 2008 existant). Voici les instructions à suivre :

# 1. Analyse la description de l'activité principale de l'entreprise et le code NACE 2008 fourni par l'utilisateur.
# 2. À partir de la liste des codes NACE 2025 disponible, identifie la catégorie la plus appropriée qui correspond à l'activité principale de l'entreprise.
# 3. Retourne le code NACE 2025 au format JSON comme spécifié par l'utilisateur. Si la description de l'activité de l'entreprise n'est pas suffisamment précise pour identifier un code NACE 2025 adéquat, retourne `null` dans le JSON.
# 4. Évalue la cohérence entre le code NACE 2008 fourni et la description de l'activité de l'entreprise. Si le code NACE 2008 ne semble pas correspondre à cette description, retourne `False` dans le champ `nace08_valid` du JSON. Note que si tu arrives à classer la description de l'activité de l'entreprise dans un code NACE 2025, le champ `nace08_valid` devrait `True`, sinon il y a incohérence.
# 5. Réponds seulement avec le JSON complété aucune autres information ne doit être retourné.
# """

# CLASSIF_PROMPT = """\
# - Activité principale de l'entreprise :
# {activity}

# - Ancien code NACE 2008 affecté:
# {nace08}

# - Liste des codes NACE potentiels et leurs notes explicatives :
# {proposed_codes}

# {format_instructions}
# """


def format_code25(codes: list):
    return "\n\n".join(
        [
            f"{nace2025.code}: {nace2025.label}\n{extract_info(nace2025, paragraphs=["include", "not_include", "notes"])}"
            for nace2025 in codes
        ]
    )


def format_code08(codes: list):
    return "\n\n".join([f"{nace08.code}: {nace08.label}" for nace08 in codes])


def extract_info(nace2025, paragraphs=["include", "not_include", "notes"]):
    info = [
        getattr(nace2025, paragraph)
        for paragraph in paragraphs
        if getattr(nace2025, paragraph) is not None
    ]
    return "\n\n".join(info) if info else ""


def generate_prompt(row, mapping, parser):
    nace08 = row.apet_finale
    activity = row.libelle_activite
    row_id = row.liasse_numero

    proposed_codes = next((m.naf2025 for m in mapping if m.code == nace08))
    prompt = CLASSIF_PROMPT.format(
        **{
            "activity": activity,
            "nace08": format_code08([next((m for m in mapping if m.code == nace08))]),
            "proposed_codes": format_code25(proposed_codes),
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return PromptData(
        id=row_id,
        proposed_codes=[c.code for c in proposed_codes],
        prompt=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
