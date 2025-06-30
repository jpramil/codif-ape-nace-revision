import argparse
import os
from datetime import datetime

import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from vllm import LLM
from vllm.sampling_params import SamplingParams

import config
from constants.llm import LLM_MODEL, MAX_NEW_TOKEN, MODEL_TO_ARGS, TEMPERATURE
from constants.paths import URL_SIRENE4_EXTRACTION
from evaluation.evaluation import calculate_accuracy, get_prompt_mapping
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from utils.data import get_ambiguous_data, get_file_system, get_ground_truth

config.setup()


def run_encode(strategy_cls, experiment_name, run_name, llm_name, third):
    strategy = strategy_cls()
    fs = get_file_system()

    data = get_ambiguous_data(third, only_annotated=True)

    prompts = strategy.get_prompts(data)
    batch_prompts = [p.prompt for p in prompts]

    local_path_model = f"{os.getenv('LOCAL_PATH')}/{llm_name}"
    llm = LLM(model=local_path_model, **MODEL_TO_ARGS.get(llm_name, {}))

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN,
        temperature=TEMPERATURE,
        seed=2025,
        logprobs=1,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        outputs = llm.chat(batch_prompts, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        logprobs = [output.outputs[0].logprobs for output in outputs]

        from src.llm.response import process_response  # Same for both

        results = [
            process_response(response, prompt, strategy.parser, logprob)
            for response, prompt, logprob in zip(responses, prompts, logprobs)
        ]

        df = pd.DataFrame(results)
        results_df = data[0] if isinstance(data, tuple) else data
        results_df = results_df.merge(df, on="liasse_numero")
        results_df = strategy.postprocess_results(results_df)

        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        pq.write_to_dataset(
            pa.Table.from_pandas(results_df),
            root_path=f"{strategy.output_path}/{'--'.join(llm_name.split('/'))}",
            partition_cols=["codable"],
            basename_template=f"part-{{i}}{f'-{third}' if third else ''}--{date}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            filesystem=fs,
        )

        # Evaluation
        ground_truth = get_ground_truth()
        prompt_mapping = get_prompt_mapping(prompts, ground_truth)
        ground_truth = ground_truth.merge(prompt_mapping, on="liasse_numero", how="inner")

        eval_df = ground_truth.merge(
            results_df[["liasse_numero", "nace2025", "codable"]], on="liasse_numero", how="inner"
        )

        accuracies = (
            calculate_accuracy(eval_df)
            | calculate_accuracy(eval_df, filter_col="mapping_ok")
            | calculate_accuracy(eval_df, filter_col="codable")
        )

        mlflow.log_params(
            {
                "LLM_MODEL": llm_name,
                "TEMPERATURE": TEMPERATURE,
                "input_path": URL_SIRENE4_EXTRACTION,
                "output_path": f"{strategy.output_path}/{'--'.join(llm_name.split('/'))}/part-{third if third else 0}--{date}.parquet",
            }
        )

        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["rag", "cag"], required=True)
    parser.add_argument("--experiment_name", type=str, default="Test")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default=LLM_MODEL, choices=MODEL_TO_ARGS.keys())
    parser.add_argument("--third", type=int, default=None)

    args = parser.parse_args()

    assert "MLFLOW_TRACKING_URI" in os.environ, "Set MLFLOW_TRACKING_URI"

    STRATEGY_MAP = {
        "rag": RAGStrategy,
        "cag": CAGStrategy,
    }

    run_encode(
        STRATEGY_MAP[args.strategy],
        args.experiment_name,
        args.run_name,
        args.llm_name,
        args.third,
    )
