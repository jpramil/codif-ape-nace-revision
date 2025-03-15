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

from src.constants.data import VAR_TO_KEEP
from src.constants.llm import (
    LLM_MODEL,
    MAX_NEW_TOKEN,
    MODEL_TO_ARGS,
    REP_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from src.constants.paths import URL_PROMPTS_RAG, URL_SIRENE4_AMBIGUOUS_RAG, URL_SIRENE4_EXTRACTION
from src.constants.vector_db import COLLECTION_NAME
from src.evaluation.evaluation import calculate_accuracy, get_prompt_mapping
from src.llm.prompting import generate_prompts_from_data, load_prompts_from_file
from src.llm.response import RAGResponse, process_response
from src.utils.data import get_ambiguous_data, get_file_system, get_ground_truth
from src.vector_db.loading import get_vector_db

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def encode_ambiguous(
    experiment_name: str,
    run_name: str,
    llm_name: str = LLM_MODEL,
    prompts_from_file: bool = True,
    third: int = None,
):
    parser = PydanticOutputParser(pydantic_object=RAGResponse)
    fs = get_file_system()

    # Get data
    data, _ = get_ambiguous_data(fs, VAR_TO_KEEP, third, only_annotated=True)

    # Get prompts
    if prompts_from_file:
        prompts = load_prompts_from_file(URL_PROMPTS_RAG, fs)
    else:
        vector_db = get_vector_db(COLLECTION_NAME)
        prompts = generate_prompts_from_data(data, parser, retriever=vector_db)

    batch_prompts = [p.prompt for p in prompts]

    # Initialize LLM
    local_path_model = f"{os.getenv('LOCAL_PATH')}/{llm_name}"
    llm = LLM(
        model=local_path_model,
        **MODEL_TO_ARGS.get(llm_name, {}),
        hf_overrides={"sliding_window": None},
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        seed=2025,
        logprobs=1,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        outputs = llm.chat(batch_prompts, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        logprobs = [output.outputs[0].logprobs for output in outputs]

        results = [
            process_response(
                response=response,
                prompt=prompt,
                parser=parser,
                logprobs=logprob,
            )
            for response, prompt, logprob in zip(responses, prompts, logprobs)
        ]

        results_df = data.merge(pd.DataFrame(results), on="liasse_numero")

        # Back to code without dot
        results_df["nace2025"] = results_df["nace2025"].str.replace(".", "", regex=False)

        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        pq.write_to_dataset(
            pa.Table.from_pandas(results_df),
            root_path=f"{URL_SIRENE4_AMBIGUOUS_RAG}/{'--'.join(llm_name.split('/'))}",
            partition_cols=["codable"],
            basename_template=f"part-{{i}}{f'-{third}' if third else ''}{f'--{date}'}.parquet",  # Filename template for Parquet parts
            existing_data_behavior="overwrite_or_ignore",
            filesystem=fs,
        )

        # EVALUATION
        ground_truth = get_ground_truth()

        prompt_mapping = get_prompt_mapping(prompts, ground_truth)

        ground_truth = ground_truth.merge(prompt_mapping, on="liasse_numero", how="inner")

        eval_df = ground_truth.merge(
            results_df[["liasse_numero", "nace2025", "codable"]], on="liasse_numero", how="inner"
        )

        accuracies_overall = calculate_accuracy(eval_df)
        accuracies_llm = calculate_accuracy(eval_df, filter_col="mapping_ok")
        accuracies_codable = calculate_accuracy(eval_df, filter_col="codable")

        # Log MLflow parameters and metrics
        mlflow.log_params(
            {
                "LLM_MODEL": llm_name,
                "TEMPERATURE": TEMPERATURE,
                "TOP_P": TOP_P,
                "REP_PENALTY": REP_PENALTY,
                "input_path": URL_SIRENE4_EXTRACTION,
                "output_path": f"{URL_SIRENE4_AMBIGUOUS_RAG}/{'--'.join(llm_name.split('/'))}/part-{third if third else 0}--{date}.parquet",
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

    assert "MLFLOW_TRACKING_URI" in os.environ, (
        "Please set the MLFLOW_TRACKING_URI environment variable."
    )

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
        "--prompts_from_file",
        type=bool,
        default=True,
        help="Whether to use prompts from file",
    )
    parser.add_argument(
        "--third",
        type=int,
        required=False,
        help="Third of the dataset to process",
    )

    args = parser.parse_args()
    encode_ambiguous(**vars(args))
