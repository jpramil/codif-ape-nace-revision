import pandas as pd
import duckdb
import os
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
from src.utils.data import get_file_system
from src.mappings.mappings import create_mapping
from src.llm.model import get_model
from src.llm.prompting import generate_prompt


def encore_multivoque(
    url_source: str,
    url_out: str,
    model_name: str,
    device: str = "cuda",
):
    """
    Processes multivoque (ambiguous) NAF codes from a source dataset and relabels them using a language model.

    Parameters:
    -----------
    url_source : str
        The S3 URL of the source dataset in Parquet format to be processed.

    url_out : str
        The S3 URL where the relabeled output dataset will be saved as a Parquet file.

    model_name : str
        The name or path of the language model to be used for generating predictions.

    device : str, optional, default="cuda"
        The device on which to run the model (e.g., 'cuda' for GPU, 'cpu' for CPU).
    """

    fs = get_file_system()

    # Load excel files containing informations about mapping
    with fs.open("s3://projet-ape/NAF-revision/table-correspondance-naf2025.xls") as f:
        table_corres = pd.read_excel(f, dtype=str)

    with fs.open("s3://projet-ape/NAF-revision/notes-explicatives-naf2025.xlsx") as f:
        notes_ex = pd.read_excel(f, dtype=str)

    mapping = create_mapping(table_corres, notes_ex)

    # Select all multivoque codes
    multivoques = [naf08 for naf08 in mapping.keys() if len(mapping[naf08]["naf25"]) > 1]

    con = duckdb.connect(database=":memory:")
    data = con.query(
        f"""
        SELECT
            *
        FROM
            read_parquet('{url_source}')
        WHERE
            apet_finale IN ('{"', '".join(multivoques)}')
    ;
    """
    ).to_df()

    # We keep only unique ids
    data = data[~data.duplicated(subset="liasse_numero")]

    # We keep only non duplicated description and complementary variables
    data = data[
        ~data.duplicated(
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
    data.reset_index(drop=True, inplace=True)
    con.close()

    tokenizer, model = get_model(model_name, device=device)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for row in tqdm(data.loc[:10, :].itertuples(), total=data.shape[0]):
        # Generate the prompt and append the eos_token (end of sequence marker)
        prompt = f"{generate_prompt(mapping, row.apet_finale, row.libelle_activite, include_notes=False)}"
        prompt += tokenizer.eos_token

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate the output
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,  # Use the eos_token_id
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode the generated output
        response = tokenizer.decode(outputs[0][-4:], skip_special_tokens=True)

        # Make sure the predicted code is from the list of potential code
        if response not in mapping[row.apet_finale]["naf25"]:
            response = None

        results.append({"id": row.id, "liasse_numero": row.liasse_numero, "naf_25_niv5": response})

    data = data.merge(pd.DataFrame(results), on=["id", "liasse_numero"])

    pq.write_table(
        pa.Table.from_pandas(data),
        url_out,
        filesystem=fs,
    )


if __name__ == "__main__":
    assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable."

    URL = "s3://projet-ape/extractions/20240812_sirene4.parquet"
    URL_OUT = "s3://projet-ape/NAF-revision/relabeled-data/20240812_sirene4_multivoques.parquet"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda"

    encore_multivoque(URL, URL_OUT, model_name, device=device)
