import os

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.constants.vector_db import QDRANT_URL

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_qdrant_client() -> QdrantClient:
    """Initialize and return the Qdrant client."""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        port=443,  # Use integer instead of string for better compatibility
        https=True,
    )


def get_embedding_model_name(client: QdrantClient, collection_name: str) -> str:
    """Retrieve the embedding model name from the Qdrant collection."""
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        return next(iter(collection_info.config.params.vectors.keys()))
    except Exception as e:
        raise RuntimeError(f"Error retrieving embedding model: {e}")


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Initialize the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


def get_reranker_model(model_name: str) -> HuggingFaceEmbeddings:
    """Initialize the HuggingFace reranker model."""
    return HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
    )


def get_vector_db(collection_name: str) -> QdrantVectorStore:
    """Initialize the Qdrant Vector Store from the existing collection."""
    client = get_qdrant_client()
    emb_model_name = get_embedding_model_name(client, collection_name)
    emb_model = get_embedding_model(emb_model_name)

    return QdrantVectorStore.from_existing_collection(
        embedding=emb_model,
        collection_name=collection_name,
        vector_name=emb_model_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        port=443,
        https=True,
    )


def get_retriever(collection_name: str, reranker_name: str):
    """Initialize the retriever from a vector database and a reranker."""
    vector_db = get_vector_db(collection_name)
    reranker = get_reranker_model(reranker_name)
    compressor = CrossEncoderReranker(model=reranker, top_n=5)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vector_db.as_retriever(search_kwargs={"k": 35})
    )
