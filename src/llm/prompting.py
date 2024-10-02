from collections import namedtuple

PromptData = namedtuple("PromptData", ["id", "proposed_codes", "prompt"])

SYS_PROMPT = """Tu es un expert de la Nomenclature statistique des Activités économiques dans la Communauté Européenne (NACE). Tu es chargé de réaliser le changement de nomenclature. Pour cela, tu dois attribuer un code NACE 2025 à une entreprise en fonction du descriptif de son activité à partir d'une liste de code proposée déduit de son code NACE 2008 connu. Voici les instructions à suivre :

1. Analyse l'activité de l'entreprise et le code NACE 2008 affecté fournit par l'utilisateur.
2. A partir de la liste des catégories d'activités fournie, identifie le code NACE 2025 le plus approprié basé sur la description de l'activité de l'entreprise.
3. Retoure le code NACE 2025 au format JSON comme spécifié par l'utilisateur. Si la description de l'entreprise n'est pas suffisament claire, pour être classée dans une des catégories proposées, retourne `null` dans le JSON.
4. Vérifie la cohérence du code NACE 2008 affecté avec la description de l'activité de l'entreprise. Si le code NACE 2008 affecté ne correspond pas à la description de l'entreprise, retourne `False` dans le champ `nace08_valid` du JSON.
"""

CLASSIF_PROMPT = """\
- Activité de l'entreprise :
{activity}

- Ancien code NACE 2008 affecté:
{nace08}

- Liste des codes NACE potentiels et leurs notes explicatives :
{proposed_codes}

{format_instructions}
"""


def format_code25(codes: list):
    return "\n\n".join(
        [
            f"{nace2025.code}: {nace2025.label}\n{extract_info(nace2025, paragraphs=["include", "not_include", "notes"])}"
            for nace2025 in codes
        ]
    )


def format_code08(codes: list):
    return "\n\n".join(
        [
            f"{nace08.code}: {nace08.label}"
            for nace08 in codes
        ]
    )


def extract_info(nace2025, paragraphs=["include", "not_include", "notes"]):
    info = [getattr(nace2025, paragraph) for paragraph in paragraphs if getattr(nace2025, paragraph) is not None]
    return "\n\n".join(info) if info else ""


def generate_prompt(row, mapping, parser):
    nace08 = row.apet_finale
    activity = row.libelle_activite
    row_id = row.id

    proposed_codes = next((m.naf2025 for m in mapping if m.code == nace08))
    prompt = CLASSIF_PROMPT.format(
            **{
                "activity": activity,
                "nace08": format_code08(next((m for m in mapping if m.code == nace08))),
                "proposed_codes": format_code25(proposed_codes),
                "format_instructions": parser.get_format_instructions(),
            }
        )
    return PromptData(
        id=row_id,
        proposed_codes=[c.code for c in proposed_codes],
        prompt= [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
        ]
    )