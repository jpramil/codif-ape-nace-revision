import argparse
import asyncio
import os

import mlflow

import config
from constants.llm import LLM_MODEL, MODEL_TO_ARGS
from constants.paths import URL_SIRENE4_EXTRACTION
from evaluation.evaluator import Evaluator
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from utils.data import get_ambiguous_data

config.setup()

# third = 1
# llm_name = "Qwen/Qwen2.5-0.5B"
# STRATEGY_MAP = {
#     "rag": RAGStrategy,
#     "cag": CAGStrategy,
# }
# strategy_cls = STRATEGY_MAP["rag"]


async def run_encode(strategy_cls, experiment_name, run_name, llm_name, third):
    strategy = strategy_cls(
        generation_model=llm_name,
    )

    data = get_ambiguous_data(strategy.mapping, third, only_annotated=True)

    data = data.head(20)
    # prompts = await strategy.get_prompts(data)
    prompts = asyncio.run(strategy.get_prompts(data))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        outputs = strategy.call_llm(prompts, strategy.sampling_params)

        processed_outputs = strategy.process_outputs(outputs)

        results = data.merge(processed_outputs, left_index=True, right_index=True)

        output_path = strategy.save_results(results, third)

        metrics = Evaluator().evaluate(results, prompts)

        # Log MLflow parameters and metrics
        mlflow.log_params(
            {
                "LLM_MODEL": llm_name,
                "TEMPERATURE": strategy.sampling_params.temperature,
                "input_path": URL_SIRENE4_EXTRACTION,
                "output_path": output_path,
                "num_coded": results["codable"].sum(),
                "num_not_coded": len(results) - results["codable"].sum(),
                "pct_not_coded": round((len(results) - results["codable"].sum()) / len(results) * 100, 2),
            }
        )

        for metric, value in metrics.items():
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

    asyncio.run(
        run_encode(
            STRATEGY_MAP[args.strategy],
            args.experiment_name,
            args.run_name,
            args.llm_name,
            args.third,
        )
    )
