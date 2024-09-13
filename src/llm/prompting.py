def generate_prompt(
    mapping: dict, code_naf08: str, activite: str, include_notes: bool = True
) -> str:
    """
    Generate a prompt for the classification task based on the NACE statistical nomenclature.

    Args:
        mapping (dict): A dictionary mapping NAF08 codes to NAF25 codes.
        code_naf08 (str): The NAF08 code of the company.
        activite (str): The activity of the company.
        include_notes (bool) : Whether including explicative notes or not.

    Returns:
        str: The NAF25 code of the company.
    """

    notes_explicatives = [
        f"""\
        {i}. Code NACE : {code}

        * Libellé du code : {details["libelle"]}\

        {f"*  {details["comprend"]} \n\n  * {details["comprend_pas"]}" if include_notes else ""}
        """
        for i, (code, details) in enumerate(mapping[code_naf08]["naf25"].items(), start=1)
    ]

    PROMPT = f"""\
Voici une tâche de classification basée sur la nomenclature statistique NACE. Votre objectif est d'analyser l'activité d'une entreprise décrite ci-dessous et de choisir, parmi une liste de codes potentiels, celui qui correspond le mieux à cette activité. Chaque code est accompagné de notes explicatives précisant les activités couvertes et celles exclues.

Activité de l'entreprise :
{activite}

Liste des codes NACE potentiels et leurs notes explicatives :
{"\n".join(notes_explicatives)}

Votre tâche est de choisir le code NACE qui correspond le plus précisément à l'activité de l'entreprise en vous basant sur les notes explicatives. Répondez uniquement avec le code NACE sélectionné, sans explication supplémentaire, parmi la liste des codes suivants : {", ".join(mapping[code_naf08]["naf25"].keys())}. Si aucun des codes de la liste ne vous semble correct répondez "ERREUR"\
"""
    return PROMPT
