import asyncio
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser
from langfuse import Langfuse
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams

from constants.paths import URL_SIRENE4_AMBIGUOUS_RAG
from constants.vector_db import COLLECTION_NAME
from constants.llm import (
    MAX_NEW_TOKEN,
    MODEL_TO_ARGS,
    TEMPERATURE,
)
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


class RAGStrategy(EncodeStrategy):
    def __init__(
        self,
        generation_model: str = "gemma3:27b",
        reranker_model: str = None,
    ):
        self.generation_model = generation_model
        self.reranker_model = reranker_model
        self.db = get_retriever(COLLECTION_NAME, self.reranker_model)
        self.prompt_template = Langfuse().get_prompt("rag-classifier", label="production")
        self.prompt_template_retriever = Langfuse().get_prompt("retriever", label="production")
        self.llm = LLM(
            model=f"{self.generation_model}",  # {os.getenv('LOCAL_PATH')/}
            **MODEL_TO_ARGS.get(self.generation_model, {}),
        )
        self.sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKEN,
            temperature=TEMPERATURE,
            seed=2025,
            logprobs=1,
            guided_decoding=GuidedDecodingParams(json=RAGResponse.model_json_schema()),
        )
        # super().__init__(client=self.llm, generation_model=self.generation_model)

    @property
    def parser(self):
        return PydanticOutputParser(pydantic_object=RAGResponse)

    async def get_prompts(self, data: pd.DataFrame):
        tasks = [self.create_prompt(row) for row in data.to_dict(orient="records")]
        return await tqdm.gather(*tasks)

    @property
    def output_path(self):
        return URL_SIRENE4_AMBIGUOUS_RAG

    def postprocess_results(self, df):
        pass

    async def create_prompt(self, row, top_k: int = 5) -> List[Dict]:
        activity = super()._format_activity_description(row)
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

    async def _call_llm(self, messages: List[Dict]) -> Optional[RAGResponse]:
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        logprobs = [output.outputs[0].logprobs for output in outputs]
        # TODO : Validate responses
        return responses, logprobs
