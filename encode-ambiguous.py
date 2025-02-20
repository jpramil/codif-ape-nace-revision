import argparse
import logging
import os
from datetime import datetime

import duckdb
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
from src.constants.prompting import MODEL_TO_PROMPT_FORMAT
from src.llm.prompting import apply_template, generate_prompt
from src.llm.response import LLMResponse, process_response
from src.mappings.mappings import get_mapping
from src.utils.data import get_file_system, load_excel_from_fs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def encore_multivoque(
    experiment_name: str,
    run_name: str,
    llm_name: str = LLM_MODEL,
    third: int = None,
):
    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    fs = get_file_system()

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

    # Load excel files containing informations about mapping
    table_corres = load_excel_from_fs(fs, URL_MAPPING_TABLE)
    notes_ex = load_excel_from_fs(fs, URL_EXPLANATORY_NOTES)

    mapping = get_mapping(notes_ex, table_corres)
    mapping_ambiguous = [code for code in mapping if len(code.naf2025) > 1]

    con = duckdb.connect(database=":memory:")
    data = (
        con.query(
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
            apet_finale IN ('{"', '".join([m.code for m in mapping_ambiguous])}')
    ;
    """
        )
        .to_df()
        .loc[:, VAR_TO_KEEP]
    )

    # We keep only unique ids
    data = data.drop_duplicates(subset="liasse_numero")

    # We keep only non duplicated description and complementary variables
    data = data.drop_duplicates(
        subset=[v for v in VAR_TO_KEEP if v != "liasse_numero" and v != "apet_finale"]
    )
    data.reset_index(drop=True, inplace=True)
    con.close()

    ground_truth = (
        pq.ParquetDataset(URL_GROUND_TRUTH.replace("s3://", ""), filesystem=fs).read().to_pandas()
    )
    ground_truth = ground_truth.drop_duplicates(subset="liasse_numero")

    # Check if the mapping is correct
    def check_mapping(naf08, naf25):
        return naf25 in naf08_to_naf2025.get(naf08, set())

    naf08_to_naf2025 = {m.code: [c.code for c in m.naf2025] for m in mapping}
    ground_truth["mapping_ok"] = [
        check_mapping(naf08, naf25)
        for naf08, naf25 in zip(ground_truth["NAF2008_code"], ground_truth["apet_manual"])
    ]

    # # TODO: Temp to only run data that has been manually coded + some random data
    # data_ground_truth = data.loc[data["liasse_numero"].isin(ground_truth["liasse_numero"].tolist())]
    # data_not_ground_truth = data.loc[
    #     ~data["liasse_numero"].isin(ground_truth["liasse_numero"].tolist())
    # ].sample(300000 - data_ground_truth.shape[0], random_state=2025)
    # data = pd.concat([data_ground_truth, data_not_ground_truth], axis=0)

    # cache_model_from_hf_hub(
    #     llm_name,
    # )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        seed=2025,
    )

    local_path_model = os.path.expanduser(f"~/.cache/huggingface/hub/{llm_name}")
    llm = LLM(model=local_path_model, **MODEL_TO_ARGS.get(llm_name, {}))

    # Sort data by liasse_numero to ensure reproducibility
    data = data.sort_values("liasse_numero").reset_index(drop=True)

    # If third is specified, process only a subset of the data for that third
    if third is not None:
        idx_for_subset = [
            ((data.shape[0] // 3) * (third - 1)),  # Start index for the subset
            ((data.shape[0] // 3) * third),  # End index for the subset
        ]
        idx_for_subset[-1] = (
            idx_for_subset[-1] if third != 3 else data.shape[0]
        )  # Adjust for the last third
        data = data.iloc[idx_for_subset[0] : idx_for_subset[1]]  # Select subset

    prompts = [generate_prompt(row, mapping_ambiguous, parser) for row in data.itertuples()]

    batch_prompts = apply_template([p.prompt for p in prompts], MODEL_TO_PROMPT_FORMAT[llm_name])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        responses = [outputs[i].outputs[0].text for i in range(len(outputs))]

        results = [
            process_response(response=response, prompt=prompt, parser=parser)
            for response, prompt in zip(responses, prompts)
        ]

        results_df = data.merge(pd.DataFrame(results), on="liasse_numero").loc[
            :,
            VAR_TO_KEEP
            + [
                "nace2025",
                "nace08_valid",
                "codable",
            ],
        ]

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

        mlflow.log_param("num_coded", results_df["codable"].sum())
        mlflow.log_param("num_not_coded", len(results_df) - results_df["codable"].sum())
        mlflow.log_param(
            "pct_not_coded",
            round((len(results_df) - results_df["codable"].sum()) / len(results_df) * 100, 2),
        )

        # EVALUATION
        ground_truth = ground_truth.loc[:, ["liasse_numero", "apet_manual", "mapping_ok"]]

        eval_df = ground_truth.merge(
            results_df[["liasse_numero", "nace2025", "codable"]],
            on="liasse_numero",
            how="inner",
        )
        mlflow.log_param("eval_size", eval_df.shape[0])

        accuracies_overall = {
            f"accuracy_overall_lvl_{i}": round(
                (eval_df["apet_manual"].str[:i] == eval_df["nace2025"].str[:i]).mean() * 100,
                2,
            )
            for i in [5, 4, 3, 2, 1]
        }

        # Accuracies when mapping is correct (true code is in the proposed list for the llm)
        mlflow.log_param("mapping_ok", eval_df["mapping_ok"].sum())
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
        for metric, value in (accuracies_overall | accuracies_llm | accuracies_codable).items():
            mlflow.log_metric(metric, value)

        mlflow.log_param("LLM_MODEL", llm_name)
        mlflow.log_param("TEMPERATURE", TEMPERATURE)
        mlflow.log_param("TOP_P", TOP_P)
        mlflow.log_param("REP_PENALTY", REP_PENALTY)
        mlflow.log_param("input_path", URL_SIRENE4_EXTRACTION)
        mlflow.log_param(
            "output_path",
            f"{URL_SIRENE4_AMBIGUOUS}/{"--".join(llm_name.split("/"))}/part-{third if third else 0}--{date}.parquet",
        )


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

    # Optional argument for specifying the third of the dataset to process
    parser.add_argument(
        "--third",
        type=int,
        required=False,
        help="Third of the dataset to process",
    )

    args = parser.parse_args()

    encore_multivoque(**vars(args))
