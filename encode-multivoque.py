import pandas as pd
from src.utils.data import get_file_system
from src.mappings.mappings import get_mapping
import duckdb
import os
import pyarrow.parquet as pq
import pyarrow as pa
from src.constants.paths import URL_SIRENE4_EXTRACTION, URL_SIRENE4_MULTIVOCAL, URL_MAPPING_TABLE, URL_EXPLANATORY_NOTES
from src.llm.response import LLMResponse, process_response
from src.constants.llm import LLM_MODEL, MAX_NEW_TOKEN, TEMPERATURE
from src.llm.model import cache_model_from_hf_hub
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams
from src.llm.prompting import generate_prompt
from langchain_core.output_parsers import PydanticOutputParser
import argparse


def encore_multivoque(
    model_name: str,
):

    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    fs = get_file_system()

    # Load excel files containing informations about mapping
    with fs.open(URL_MAPPING_TABLE) as f:
        table_corres = pd.read_excel(f, dtype=str)

    with fs.open(URL_EXPLANATORY_NOTES) as f:
        notes_ex = pd.read_excel(f, dtype=str)

    mapping = get_mapping(notes_ex, table_corres)
    mapping_multivocal = [code for code in mapping if len(code.naf2025) > 1]

    con = duckdb.connect(database=":memory:")
    data = con.query(
        f"""
        SET s3_endpoint='{os.getenv("AWS_S3_ENDPOINT")}';
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';

        SELECT
            *
        FROM
            read_parquet('{URL_SIRENE4_EXTRACTION}')
        WHERE
            apet_finale IN ('{"', '".join([m.code for m in mapping_multivocal])}')
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

    cache_model_from_hf_hub(
        LLM_MODEL,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN, temperature=TEMPERATURE, top_p=0.8, repetition_penalty=1.05
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = LLM(model=LLM_MODEL, max_model_len=20000, gpu_memory_utilization=0.95)

    prompts = [
        generate_prompt(row, mapping_multivocal, parser)
        for row in data.head(10).itertuples()
    ]

    batch_prompts = tokenizer.apply_chat_template(
    [p.prompt for p in prompts], tokenize=False, add_generation_prompt=True
    )
    outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
    responses = [outputs[i].outputs[0].text for i in range(len(outputs))]

    results = [process_response(response=response, prompt=prompt, parser=parser) for response, prompt in  zip(responses, prompts)]

    pq.write_to_dataset(
        pa.Table.from_pylist(results),
        root_path=f"{URL_SIRENE4_MULTIVOCAL}/{LLM_MODEL}",
        partition_cols=["nace08_valid", "codable"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recode into NACE2025 nomenclature")

    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model name"
    )

    args = parser.parse_args()
    encore_multivoque(model_name=args.model_name)
