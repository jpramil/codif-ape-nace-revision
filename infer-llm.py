import pandas as pd
import unicodedata
import duckdb
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pyarrow.parquet as pq
import s3fs
import pyarrow as pa


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

Votre tâche est de choisir le code NACE qui correspond le plus précisément à l'activité de l'entreprise en vous basant sur les notes explicatives. Répondez uniquement avec le code NACE sélectionné, sans explication supplémentaire, parmi la liste des codes suivants : {", ".join(mapping[code_naf08]["naf25"].keys())}. Si aucun des codes de la liste ne vous semble correct répondez "ERREUR"\
"""
    return PROMPT


def create_mapping(mapping_table: pd.DataFrame, explanatory_notes: pd.DataFrame) -> dict:
    """
    Processes the provided `mapping_table` and `explanatory_notes` DataFrames to create a
    hierarchical mapping between NAF 2008 and NAF 2025 codes, and enriches the mapping
    with additional explanatory notes.

    Parameters:
    ----------
    mapping_table : pd.DataFrame
        The DataFrame containing the correspondence between old and new NAF codes.
        Expected columns are: NAF 2008 codes, NAF 2025 codes, and their descriptions.

    explanatory_notes : pd.DataFrame
        The DataFrame containing explanatory notes for the NAF codes, including
        general notes, what the code comprises, and what it does not comprise.

    Returns:
    -------
    mapping : dict
        A nested dictionary mapping NAF 2008 codes (`naf08_niv5`) to:
        - "libelle" : The description of the NAF 2008 code.
        - "naf25" : A dictionary mapping NAF 2025 codes (`naf25_niv5`) to:
            - "libelle" : The description of the NAF 2025 code.
            - "notes" : General notes on the NAF 2025 code.
            - "comprend" : What the NAF 2025 code includes.
            - "comprend_pas" : What the NAF 2025 code does not include.
    """

    columns_mapping = {
        "NAFold-code\n(code niveau sous-classe de la nomenclature actuelle)": "naf08_niv5",
        "NACEold-code\n(niveau classe)": "naf08_niv4",
        "NAFold-intitulé\n(niveau sous-classe)": "lib_naf08_niv5",
        "NACEnew-code\n(niveau classe)": "naf25_niv4",
        "NAFnew-code\n(code niveau sous-classe de la nomenclature 2025, correspondance logique avec les NAFold-codes)": "naf25_niv5",
        "NAFnew-intitulé\n(niveau sous-classe)": "lib_naf25_niv5",
    }

    mapping_table = (
        mapping_table.iloc[:, [1, 3, 2, 10, 5, 11]]
        .rename(columns=columns_mapping)
        .assign(
            naf08_niv5=mapping_table.iloc[:, 1].str.replace(".", "", regex=False),
            naf08_niv4=mapping_table.iloc[:, 3].str.replace(".", "", regex=False),
            naf25_niv4=mapping_table.iloc[:, 5].str.replace(".", "", regex=False),
            naf25_niv5=mapping_table.iloc[:, 10].str.replace(".", "", regex=False),
        )
    )

    mapping = {
        code08: {
            "libelle": subset["lib_naf08_niv5"].iloc[0],
            "naf25": {
                row["naf25_niv5"]: {"libelle": row["lib_naf25_niv5"]}
                for _, row in subset.iterrows()
            },
        }
        for code08, subset in mapping_table.groupby("naf08_niv5")
    }

    explanatory_notes = (
        explanatory_notes.iloc[:, [6, 9, 10, 11, 12, 13, 7, 8]]
        .rename(
            columns={
                "Code.NACE.Rev.2.1": "naf08_niv4",
                "Titre": "lib_naf08_niv5",
                "Note.générale": "notes",
            }
        )
        .assign(naf08_niv4=explanatory_notes["Code.NACE.Rev.2.1"].str.replace(".", "", regex=False))
    )

    # Filling `mapping` with notes information
    for code08, code08_data in mapping.items():
        for code25, code25_data in code08_data["naf25"].items():
            # Attempt to find level 5 naf code in explanatory_notes
            row = explanatory_notes.loc[explanatory_notes["naf08_niv4"] == code25]

            # If doesn't exist, try to find level 4 naf code in explanatory_notes
            if row.empty:
                row = explanatory_notes.loc[explanatory_notes["naf08_niv4"] == code25[:-1]]
                if row.shape[0] != 1:
                    row = explanatory_notes.loc[
                        (explanatory_notes["naf08_niv4"] == code25[:-1])
                        & (explanatory_notes["indic.NAF"] == "1")
                    ]
                    if row.shape[0] != 1:
                        raise ValueError(f"Could not find notes for {code25}")

            # Extract relevant fields and normalize values
            notes = normalize_value(row["notes"].iloc[0])
            comprend = normalize_value(row["Comprend"].iloc[0])
            comprend_aussi = normalize_value(row["Comprend.aussi"].iloc[0])
            comprend_pas = normalize_value(row["Ne.comprend.pas"].iloc[0])

            # Combine 'comprend' and 'comprend_aussi' if they exist
            comprend_all = (
                "\n".join(filter(None, [comprend, comprend_aussi]))
                if comprend or comprend_aussi
                else None
            )

            # Update the `mapping` dictionary with the new information
            mapping[code08]["naf25"][code25].update(
                {
                    "notes": notes,
                    "comprend": comprend_all,
                    "comprend_pas": comprend_pas,
                }
            )

    return mapping


def get_model(model_name: str, device: str = "cuda") -> tuple:
    """
    Initializes a HuggingFace tokenizer and model.

    Parameters:
    ----------
    model_name : str
        The name or path of the pre-trained model to load from the HuggingFace Hub.
        For example, 'meta-llama/Meta-Llama-3.1-8B-Instruct'.

    device : str, optional, default="cuda"
        The device on which to load the model. Typically, this will be 'cuda' for GPUs or 'cpu' for CPU execution.
    Returns:
    -------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the specified model, used for encoding and decoding text.

    model : transformers.PreTrainedModel
        The pre-trained causal language model loaded on the specified device.
    """

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)

    return tokenizer, model


table_corres = pd.read_excel("table-correspondance-naf2025.xls", dtype=str)
notes_ex = pd.read_excel("notes-explicatives-naf2025.xlsx", dtype=str)

URL = "s3://projet-ape/extractions/20240812_sirene4.parquet"
URL_OUT = "s3://projet-ape/NAF-revision/relabeled-data/20240812_sirene4_multivoques.parquet"

mapping = create_mapping(table_corres, notes_ex)

# Select all multivoque codes
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

# We keep only unique ids
data_multivoques = data_multivoques[~data_multivoques.duplicated(subset="liasse_numero")]

# We keep only non duplicated description and complementary variables
data_multivoques = data_multivoques[
    ~data_multivoques.duplicated(
        subset=[
            "apet_finale",
            "libelle_activite",
            "evenement_type",
            "cj",
            "activ_nat_et",
            "liasse_type",
            "activ_surf_et",
        ]
    )
]
data_multivoques.reset_index(drop=True, inplace=True)
con.close()

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = "cuda"

tokenizer, model = get_model(model_name, device=device)

data = data_multivoques

results = []
for row in tqdm(data.itertuples()):
    # Generate the prompt and append the eos_token (end of sequence marker)
    prompt = (
        f"{generate_prompt(mapping, row.apet_finale, row.libelle_activite, include_notes=False)}"
    )
    prompt += tokenizer.eos_token

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
    )

    # Decode the generated output, skipping special tokens
    response = tokenizer.decode(outputs[0][-4:], skip_special_tokens=True)

    # Make sure the predicted code is from the list of potential code
    if response not in mapping[row.apet_finale]["naf25"]:
        response = None

    results.append({"id": row.id, "liasse_numero": row.liasse_numero, "naf_25_niv5": response})

data = data.merge(pd.DataFrame(results), on=["id", "liasse_numero"])

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN"),
)

pq.write_table(
    pa.Table.from_pandas(data),
    URL_OUT,
    filesystem=fs,
)
