import logging
import os

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.constants.paths import URL_EXPLANATORY_NOTES, URL_MAPPING_TABLE
from src.constants.vector_db import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.mappings.mappings import get_mapping, get_nace2025_from_mapping
from src.utils.data import get_file_system, load_excel_from_fs
from src.vector_db.parsing import create_content_vdb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(collection_name: str):
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

    # Create Qdrant DB
    QdrantVectorStore.from_documents(
        document_list,
        emb_model,
        collection_name=collection_name,
        vector_name=EMBEDDING_MODEL,
        url="projet-ape-qdrant.user.lab.sspcloud.fr",
        api_key=os.getenv("QDRANT_API_KEY"),
        port="443",
        https=True,
    )

    logging.info("Qdrant DB has been created in collection '{collection_name}'.")


if __name__ == "__main__":
    main(collection_name=COLLECTION_NAME, api_key=os.getenv("QDRANT_API_KEY"))
