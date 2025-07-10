# Not UP-TO-DATE
# Implement a way to write prompts on S3 if desired in the stagregies classes instead of this script

import logging

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser

from src.constants.data import VAR_TO_KEEP
from src.constants.paths import URL_PROMPTS_RAG
from src.constants.vector_db import (
    COLLECTION_NAME,
    RERANKER_MODEL,
)
from src.evaluation.evaluation import get_prompt_mapping
from src.llm.prompting import generate_prompt
from src.llm.response import RAGResponse
from src.utils.data import get_ambiguous_data, get_file_system, get_ground_truth
from src.vector_db.loading import get_retriever

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(collection_name: str):
    parser = PydanticOutputParser(pydantic_object=RAGResponse)
    fs = get_file_system()

    # Get data
    data, _ = get_ambiguous_data(fs, VAR_TO_KEEP, third=None, only_annotated=True)

    # Get retriever
    retriever = get_retriever(COLLECTION_NAME, RERANKER_MODEL)

    # Generate prompts
    prompts = [generate_prompt(row, parser, retriever=retriever) for row in data.itertuples()]

    # Save prompts
    df = pd.DataFrame(prompts, columns=["liasse_numero", "proposed_codes", "prompt", "system_prompt"])
    df.to_parquet(URL_PROMPTS_RAG, filesystem=fs)

    ground_truth = get_ground_truth()

    prompt_mapping = get_prompt_mapping(prompts, ground_truth)

    pct_mapping_ok = prompt_mapping["mapping_ok"].sum() / len(prompt_mapping) * 100

    logging.info(f"Prompts saved to {URL_PROMPTS_RAG} with {pct_mapping_ok:.2f}% mapping ok")


if __name__ == "__main__":
    main(collection_name=COLLECTION_NAME)
