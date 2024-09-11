import pandas as pd
import unicodedata
import duckdb
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def normalize_value(value):
    return unicodedata.normalize("NFKC", value) if pd.notna(value) else None


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

Votre tâche est de choisir le code NACE qui correspond le plus précisément à l'activité de l'entreprise en vous basant sur les notes explicatives. Répondez uniquement avec le code NACE sélectionné, sans explication supplémentaire, parmi la liste des codes suivants : {", ".join(mapping[code_naf08]["naf25"].keys())}\
"""
    return PROMPT


table_corres = pd.read_excel("table-correspondance-naf2025.xls", dtype=str)
notes_ex = pd.read_excel("notes-explicatives-naf2025.xlsx", dtype=str)

URL = "s3://projet-ape/extractions/20240812_sirene4.parquet"
URL_OUT = "s3://projet-ape/extractions/20240812_sirene4_univoques.parquet"

table_corres = (
    table_corres.iloc[:, [1, 3, 2, 10, 5, 11]]
    .rename(
        columns={
            "NAFold-code\n(code niveau sous-classe de la nomenclature actuelle)": "naf08_niv5",
            "NACEold-code\n(niveau classe)": "naf08_niv4",
            "NAFold-intitulé\n(niveau sous-classe)": "lib_naf08_niv5",
            "NACEnew-code\n(niveau classe)": "naf25_niv4",
            "NAFnew-code\n(code niveau sous-classe de la nomenclature 2025, correspondance logique avec les NAFold-codes)": "naf25_niv5",
            "NAFnew-intitulé\n(niveau sous-classe)": "lib_naf25_niv5",
        }
    )
    .apply(lambda col: col.str.replace(".", "", regex=False) if col.dtype == "object" else col)
)

mapping = {
    code08: {
        "libelle": pd.unique(subset["lib_naf08_niv5"])[0],
        "naf25": {
            row["naf25_niv5"]: {"libelle": row["lib_naf25_niv5"]} for _, row in subset.iterrows()
        },
    }
    for code08, subset in table_corres.groupby("naf08_niv5")
}

notes_ex["Code.NACE.Rev.2.1"] = notes_ex["Code.NACE.Rev.2.1"].str.replace(".", "", regex=False)
notes_ex = notes_ex.iloc[:, [6, 9, 10, 11, 12, 13, 7, 8]].rename(
    columns={
        "Code.NACE.Rev.2.1": "naf08_niv4",
        "Titre": "lib_naf08_niv5",
        "Note.générale": "notes",
    }
)

for code08 in mapping.keys():
    for code25 in mapping[code08]["naf25"].keys():
        row = notes_ex.loc[notes_ex["naf08_niv4"] == code25]

        if row.empty:
            row = notes_ex.loc[notes_ex["naf08_niv4"] == code25[:-1]]
            if row.shape[0] != 1:
                row = notes_ex.loc[
                    (notes_ex["naf08_niv4"] == code25[:-1]) & (notes_ex["indic.NAF"] == "1")
                ]
                if row.shape[0] != 1:
                    raise ValueError(f"Could not find notes for {code25}")

        notes = normalize_value(row.notes.iloc[0])
        comprend = normalize_value(row.Comprend.iloc[0])
        comprend_aussi = normalize_value(row["Comprend.aussi"].iloc[0])
        comprend_pas = normalize_value(row["Ne.comprend.pas"].iloc[0])

        comprend_all = (
            "\n".join(filter(None, [comprend, comprend_aussi]))
            if any([comprend, comprend_aussi])
            else None
        )

        notes_expli = {
            "notes": notes,
            "comprend": comprend_all,
            "comprend_pas": comprend_pas,
        }
        mapping[code08]["naf25"][code25].update(notes_expli)


multivoques = [naf08 for naf08 in mapping.keys() if len(mapping[naf08]["naf25"]) > 1]

con = duckdb.connect(database=":memory:")

data_multivoques = con.query(
    f"""
    SELECT
        *
    FROM
        read_parquet('{URL}')
    WHERE
        apet_finale IN ('{"', '".join(multivoques)}')
;
"""
).to_df()

con.close()


torch.cuda.empty_cache()
device = "cuda"

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HF_TOKEN")).to(device)

# Example row to generate the prompt
row = data_multivoques.loc[1019, :]

# Generate the prompt and append the eos_token (end of sequence marker)
prompt = f"{generate_prompt(mapping, row.apet_finale, row.libelle_activite, include_notes=False)}"
prompt += tokenizer.eos_token  # Add the end-of-sequence token to the prompt

# Tokenize the input and move it to the device (GPU)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate the output from the model with eos_token_id marked
outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    temperature=0.1,
    top_p=0.8,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,  # Use the eos_token_id
    eos_token_id=tokenizer.eos_token_id,
)  # Force the model to use eos_token

# Decode the generated output, skipping special tokens
response = tokenizer.decode(outputs[0][-4:], skip_special_tokens=True)

# Print the final response, which should be just the code
print(response)
