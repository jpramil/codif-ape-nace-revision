import logging

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser

from src.constants.data import VAR_TO_KEEP
from src.constants.paths import URL_PROMPTS_RAG
from src.constants.vector_db import (
    COLLECTION_NAME,
)
from src.llm.prompting import generate_prompt
from src.llm.response import RAGResponse
from src.utils.data import get_ambiguous_data, get_file_system
from src.vector_db.loading import get_vector_db

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(collection_name: str):
    parser = PydanticOutputParser(pydantic_object=RAGResponse)
    fs = get_file_system()

    # Get data
    data, _ = get_ambiguous_data(fs, VAR_TO_KEEP, third=None, only_annotated=True)

    # Get vector db
    vector_db = get_vector_db(COLLECTION_NAME)

    # Generate prompts
    prompts = [generate_prompt(row, parser, retriever=vector_db) for row in data.itertuples()]

    # Save prompts
    df = pd.DataFrame(
        prompts, columns=["liasse_numero", "proposed_codes", "prompt", "system_prompt"]
    )
    df.to_parquet(URL_PROMPTS_RAG, filesystem=fs)

    logging.info(f"Prompts saved to {URL_PROMPTS_RAG}")


if __name__ == "__main__":
    main(collection_name=COLLECTION_NAME)
