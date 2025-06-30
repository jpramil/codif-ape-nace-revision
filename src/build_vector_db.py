import logging

from langchain_community.document_loaders import DataFrameLoader

import config
from constants.vector_db import COLLECTION_NAME
from vector_db.loading import create_vector_db, get_embedding_model
from vector_db.notices_nace import fetch_nace2025_labels

config.setup()
logger = logging.getLogger(__name__)


def main(collection_name: str):
    labels = fetch_nace2025_labels()

    # Load documents
    docs = DataFrameLoader(labels, page_content_column="content").load()

    # Initialize embedding model
    emb_model = get_embedding_model("Qwen/Qwen3-Embedding-8B")

    _ = create_vector_db(docs, emb_model, collection_name)

    logging.info(f"Qdrant DB has been created in collection '{collection_name}'.")


if __name__ == "__main__":
    main(collection_name=COLLECTION_NAME)
