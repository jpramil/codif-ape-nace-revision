# SEARCH_ALGO="similarity"
# MAX_CODE_RETRIEVED=5
# retriever = db.as_retriever(search_type=SEARCH_ALGO, search_kwargs={"k": 10})
# results = retriever.invoke(input_txt.format(task_description=task_description, query=activity))
# [r.metadata["code"] for r in results]

import logging
import os
import subprocess

import pandas as pd
from chromadb.config import Settings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.constants.paths import URL_EXPLANATORY_NOTES, URL_MAPPING_TABLE
from src.constants.vector_db import (
    CHROMA_DB_LOCAL_DIRECTORY,
    CHROMA_DB_S3_DIRECTORY,
    EMBEDDING_MODEL,
)
from src.mappings.mappings import get_mapping, get_nace2025_from_mapping
from src.utils.data import get_file_system, load_excel_from_fs
from src.vector_db.parsing import create_content_vdb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    fs = get_file_system()

    # Load data
    table_corres = load_excel_from_fs(fs, URL_MAPPING_TABLE)
    notes_ex = load_excel_from_fs(fs, URL_EXPLANATORY_NOTES)

    # Generate mapping and codes
    mapping = get_mapping(notes_ex, table_corres)
    codes_naf2025 = get_nace2025_from_mapping(mapping)

    # Create DataFrame and content
    df_naf2025 = pd.DataFrame([c.__dict__ for c in codes_naf2025])
    df_naf2025 = create_content_vdb(df_naf2025)

    # Load documents
    document_list = DataFrameLoader(df_naf2025, page_content_column="content").load()

    # Initialize embedding model
    emb_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )

    # Create and persist Chroma DB
    Chroma.from_documents(
        collection_name="labels_embeddings",
        documents=document_list,
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        embedding=emb_model,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    logging.info("Chroma DB created and persisted locally.")

    # Copy the vector database to S3
    hash_chroma = next(
        entry
        for entry in os.listdir(CHROMA_DB_LOCAL_DIRECTORY)
        if os.path.isdir(os.path.join(CHROMA_DB_LOCAL_DIRECTORY, entry))
    )
    logging.info(f"Uploading Chroma DB ({hash_chroma}) to S3")
    cmd = [
        "mc",
        "cp",
        "-r",
        f"{CHROMA_DB_LOCAL_DIRECTORY}/",
        f"s3/{CHROMA_DB_S3_DIRECTORY}/{EMBEDDING_MODEL}/{hash_chroma}/",
    ]
    with open("/dev/null", "w") as devnull:
        subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)


if __name__ == "__main__":
    main()
