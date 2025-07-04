import argparse
import asyncio
import os

import pandas as pd
from langfuse.openai import AsyncOpenAI

import config
from constants.llm import LLM_MODEL, MODEL_TO_ARGS
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from utils.data import get_ambiguous_data

config.setup()


def run_encode(strategy_cls, experiment_name, run_name, llm_name, third):
    # strategy = strategy_cls()
    strategy = RAGStrategy(
        generation_model="Qwen/Qwen2.5-0.5B",
    )
    # fs = get_file_system()
    third = 1
    data, _ = get_ambiguous_data(third, only_annotated=True)

    data = data.head(10)

    # prompts = await strategy.get_prompts(data)
    prompts = asyncio.run(strategy.get_prompts(data))

    # results = await strategy.call_llm_batch(prompts)
    results = strategy._call_llm(prompts)

    df = pd.DataFrame.from_records([r.model_dump() if r else None for r in results])

    return df


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
