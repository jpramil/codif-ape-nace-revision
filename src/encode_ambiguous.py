import argparse
import asyncio
import os

from langfuse.openai import AsyncOpenAI

import config
from constants.llm import LLM_MODEL, MODEL_TO_ARGS
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from utils.data import get_ambiguous_data

config.setup()
llm_client = AsyncOpenAI(
    base_url="https://vllm-generation.user.lab.sspcloud.fr/v1",  # "https://llm.lab.sspcloud.fr/api",
    api_key=os.environ["OPENAI_API_KEY"],
)


def run_encode(strategy_cls, experiment_name, run_name, llm_name, third):
    # strategy = strategy_cls()
    strategy = RAGStrategy(
        llm_client=llm_client,
        generation_model="mistralai/Mistral-Small-24B-Instruct-2501",  # "gemma3:27b",
    )
    # fs = get_file_system()
    third = 1
    data, _ = get_ambiguous_data(third, only_annotated=True)

    data = data.head(10)

    # prompts = await strategy.get_prompts(data)
    prompts = asyncio.run(strategy.get_prompts(data))

    # results = await strategy.call_llm_batch(prompts)
    results = asyncio.run(strategy.call_llm_batch(prompts))
    return results


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
