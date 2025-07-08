import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.schema import Document
from langfuse import Langfuse
from pydantic import BaseModel, Field, model_validator
from tqdm.asyncio import tqdm
from vllm import LLM
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

from constants.llm import (
    MAX_NEW_TOKEN,
    MODEL_TO_ARGS,
    TEMPERATURE,
)
from constants.paths import URL_SIRENE4_AMBIGUOUS_RAG
from constants.vector_db import COLLECTION_NAME
from utils.data import fetch_mapping, get_file_system
from vector_db.loading import get_retriever

from .base import EncodeStrategy

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide classification code, False otherwise."""
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code Empty if codable=False.""",
        default=None,
    )

    confidence: Optional[float] = Field(
        description="""Confidence score for the NACE2025 code, based on log probabilities. Rounded to 2 decimal places maximum.""",
        default=0.0,
    )

    @model_validator(mode="after")
    def check_nace2025_if_codable(self) -> BaseModel:
        if self.codable and not self.nace2025:
            raise ValueError("If codable=True, then nace2025 must not be None or empty.")
        return self


class RAGStrategy(EncodeStrategy):
    def __init__(
        self,
        generation_model: str = "gemma3:27b",
        reranker_model: str = None,
    ):
        self.fs = get_file_system()
        self.mapping = fetch_mapping()
        self.response_format = RAGResponse
        self.generation_model = generation_model
        self.reranker_model = reranker_model
        self.db = get_retriever(COLLECTION_NAME, self.reranker_model)
        self.prompt_template = Langfuse().get_prompt("rag-classifier", label="production")
        self.prompt_template_retriever = Langfuse().get_prompt("retriever", label="production")
        self.llm = LLM(
            model=f"{self.generation_model}",  # {os.getenv('LOCAL_PATH')/}
            **MODEL_TO_ARGS.get(self.generation_model, {}),
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKEN,
            temperature=TEMPERATURE,
            seed=2025,
            logprobs=1,
            guided_decoding=GuidedDecodingParams(json=self.response_format.model_json_schema()),
        )

    async def get_prompts(self, data: pd.DataFrame):
        tasks = [self.create_prompt(row) for row in data.to_dict(orient="records")]
        return await tqdm.gather(*tasks)

    @property
    def output_path(self):
        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        return f"{URL_SIRENE4_AMBIGUOUS_RAG}/{self.generation_model}/part-{{i}}-{{third}}--{date}.parquet"

    # TODO: implement a method that create prompt and saves it to s3 in parquet and load it back when specified
    async def create_prompt(self, row: Dict[str, Any], top_k: int = 5) -> Dict:
        activity = self._format_activity_description(row)
        query = self.prompt_template_retriever.compile(
            activity_description=activity,
        )
        docs = await self.db.asimilarity_search(query, k=top_k)
        proposed_codes, list_codes = self._format_documents(docs)

        return self.prompt_template.compile(
            activity=activity,
            proposed_codes=proposed_codes,
            list_proposed_codes=list_codes,
        )

    def _format_documents(self, docs: List[Document]) -> (str, str):
        proposed_codes = "\n\n".join(f"========\n{doc.page_content}" for doc in docs)
        list_codes = ", ".join(f"'{doc.metadata['code']}'" for doc in docs)
        return proposed_codes, list_codes
