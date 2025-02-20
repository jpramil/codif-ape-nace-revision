import argparse
import logging
import os
from datetime import datetime

import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser
from vllm import LLM
from vllm.sampling_params import SamplingParams

from src.constants.llm import (
    LLM_MODEL,
    MAX_NEW_TOKEN,
    MODEL_TO_ARGS,
    REP_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from src.constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_GROUND_TRUTH,
    URL_MAPPING_TABLE,
    URL_SIRENE4_AMBIGUOUS,
    URL_SIRENE4_EXTRACTION,
)
from src.llm.prompting import generate_prompt
from src.llm.response import LLMResponse, process_response
from src.mappings.mappings import check_mapping, get_mapping
from src.utils.data import get_file_system, load_data_from_s3, load_excel_from_fs, process_subset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VAR_TO_KEEP = [
    "liasse_numero",
    "apet_finale",
    "libelle",
    "evenement_type",
    "cj",
    "activ_nat_et",
    "liasse_type",
    "activ_surf_et",
    "activ_sec_agri_et",
    "activ_nat_lib_et",
    "activ_perm_et",
]


def encode_ambiguous(
    experiment_name: str,
    run_name: str,
    llm_name: str = LLM_MODEL,
    third: int = None,
):
    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    fs = get_file_system()

    # Load excel files containing informations about mapping
    table_corres = load_excel_from_fs(fs, URL_MAPPING_TABLE)
    notes_ex = load_excel_from_fs(fs, URL_EXPLANATORY_NOTES)
    mapping = get_mapping(notes_ex, table_corres)
    mapping_ambiguous = [code for code in mapping if len(code.naf2025) > 1]

    # Load main data
    ambiguous_codes = "', '".join([m.code for m in mapping_ambiguous])
    query = f"""
        SELECT * FROM read_parquet('{URL_SIRENE4_EXTRACTION}')
        WHERE apet_finale IN ('{ambiguous_codes}')
    """
    data = load_data_from_s3(query).loc[:, VAR_TO_KEEP].drop_duplicates(subset="liasse_numero")

    # We keep only non duplicated description and complementary variables
    data = (
        data.drop_duplicates(
            subset=[v for v in VAR_TO_KEEP if v != "liasse_numero" and v != "apet_finale"]
        )
        .sort_values("liasse_numero")
        .reset_index(drop=True)
    )

    # Load Ground Truth data
    query = f"""SELECT * FROM read_parquet('{URL_GROUND_TRUTH}')"""
    ground_truth = (
        load_data_from_s3(query)
        .loc[:, ["liasse_numero", "apet_manual", "NAF2008_code"]]
        .drop_duplicates(subset="liasse_numero")
        .sort_values("liasse_numero")
        .reset_index(drop=True)
    )

    # Check if the mapping is correct
    naf08_to_naf2025 = {m.code: [c.code for c in m.naf2025] for m in mapping}
    ground_truth["mapping_ok"] = [
        check_mapping(naf08, naf25, naf08_to_naf2025)
        for naf08, naf25 in zip(ground_truth["NAF2008_code"], ground_truth["apet_manual"])
    ]

    # # TODO: Temp to only run data that has been manually coded + some random data
    # data_ground_truth = data.loc[data["liasse_numero"].isin(ground_truth["liasse_numero"].tolist())]
    # data_not_ground_truth = data.loc[
    #     ~data["liasse_numero"].isin(ground_truth["liasse_numero"].tolist())
    # ].sample(300000 - data_ground_truth.shape[0], random_state=2025)
    # data = pd.concat([data_ground_truth, data_not_ground_truth], axis=0)

    # Initialize LLM
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        seed=2025,
    )
    local_path_model = os.path.expanduser(f"~/.cache/huggingface/hub/{llm_name}")
    llm = LLM(model=local_path_model, **MODEL_TO_ARGS.get(llm_name, {}))

    # Process data subset
    data = process_subset(data, third)
    data = data.iloc[:1000]
    # Generate prompts
    prompts = [generate_prompt(row, mapping_ambiguous, parser) for row in data.itertuples()]
    batch_prompts = [p.prompt for p in prompts]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        outputs = llm.chat(batch_prompts, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        results = [
            process_response(response=response, prompt=prompt, parser=parser)
            for response, prompt in zip(responses, prompts)
        ]

        results_df = data.merge(pd.DataFrame(results), on="liasse_numero")

        # Fill missing values with undefined for nace08 for parquet partition compatibility
        results_df["nace08_valid"] = results_df["nace08_valid"].fillna("undefined").astype(str)

        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        pq.write_to_dataset(
            pa.Table.from_pandas(results_df),
            root_path=f"{URL_SIRENE4_AMBIGUOUS}/{"--".join(llm_name.split("/"))}",
            partition_cols=["nace08_valid", "codable"],
            basename_template=f"part-{{i}}{f'-{third}' if third else ""}{f'--{date}'}.parquet",  # Filename template for Parquet parts
            existing_data_behavior="overwrite_or_ignore",
            filesystem=fs,
        )

        # EVALUATION
        ground_truth = ground_truth.loc[:, ["liasse_numero", "apet_manual", "mapping_ok"]]

        eval_df = ground_truth.merge(
            results_df[["liasse_numero", "nace2025", "codable"]],
            on="liasse_numero",
            how="inner",
        )

        accuracies_overall = {
            f"accuracy_overall_lvl_{i}": round(
                (eval_df["apet_manual"].str[:i] == eval_df["nace2025"].str[:i]).mean() * 100,
                2,
            )
            for i in [5, 4, 3, 2, 1]
        }

        accuracies_llm = {
            f"accuracy_llm_lvl_{i}": round(
                (
                    eval_df[eval_df["mapping_ok"]]["apet_manual"].str[:i]
                    == eval_df[eval_df["mapping_ok"]]["nace2025"].str[:i]
                ).mean()
                * 100,
                2,
            )
            for i in [5, 4, 3, 2, 1]
        }

        accuracies_codable = {
            f"accuracy_codable_lvl_{i}": round(
                (
                    eval_df[eval_df["codable"]]["apet_manual"].str[:i]
                    == eval_df[eval_df["codable"]]["nace2025"].str[:i]
                ).mean()
                * 100,
                2,
            )
            for i in [5, 4, 3, 2, 1]
        }

        # Log MLflow parameters and metrics
        mlflow.log_params(
            {
                "LLM_MODEL": llm_name,
                "TEMPERATURE": TEMPERATURE,
                "TOP_P": TOP_P,
                "REP_PENALTY": REP_PENALTY,
                "input_path": URL_SIRENE4_EXTRACTION,
                "output_path": f"{URL_SIRENE4_AMBIGUOUS}/{"--".join(llm_name.split("/"))}/part-{third if third else 0}--{date}.parquet",
                "num_coded": results_df["codable"].sum(),
                "num_not_coded": len(results_df) - results_df["codable"].sum(),
                "pct_not_coded": round(
                    (len(results_df) - results_df["codable"].sum()) / len(results_df) * 100, 2
                ),
                "eval_size": eval_df.shape[0],
                "mapping_ok": eval_df["mapping_ok"].sum(),
            }
        )

        for metric, value in (accuracies_overall | accuracies_llm | accuracies_codable).items():
            mlflow.log_metric(metric, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recode into NACE2025 nomenclature")

    assert (
        "MLFLOW_TRACKING_URI" in os.environ
    ), "Please set the MLFLOW_TRACKING_URI environment variable."

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Test",
        help="Experiment name in MLflow",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name in MLflow",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default=LLM_MODEL,
        help="LLM model name",
        choices=MODEL_TO_ARGS.keys(),
    )
    parser.add_argument(
        "--third",
        type=int,
        required=False,
        help="Third of the dataset to process",
    )

    args = parser.parse_args()
    encode_ambiguous(**vars(args))
