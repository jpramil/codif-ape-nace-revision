import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from langchain.schema import Document
from langfuse import Langfuse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator
from tqdm.asyncio import tqdm
from vllm import LLM
from vllm.outputs import RequestOutput
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
        # super().__init__(client=self.llm, generation_model=self.generation_model)

    async def get_prompts(self, data: pd.DataFrame):
        tasks = [self.create_prompt(row) for row in data.to_dict(orient="records")]
        return await tqdm.gather(*tasks)

    @property
    def output_path(self):
        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        template = "{url_data}/{generation_model}/part-{i}-{third}--{date}.parquet"
        return template.format(
            url_data=URL_SIRENE4_AMBIGUOUS_RAG,
            generation_model=self.generation_model,
            date=date,
            third="{third}",
            i="{i}",
        )

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

    def _call_llm(self, messages: List[Dict]) -> List[Optional[BaseModel]]:
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        return outputs

    # a mettre en super()
    def _parse_content(self, response_format: BaseModel, content: str) -> BaseModel:
        try:
            return TypeAdapter(response_format).validate_json(content)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return None

    def _process_output(self, output: RequestOutput, response_format: BaseModel) -> BaseModel:
        """
        Process the outputs from the LLM and return a list of BaseModel objects.
        """
        parsed = self._parse_content(response_format, output.outputs[0].text)
        if parsed is None:
            return response_format(codable=False, nace2025=None, confidence=0.0)

        # We get the tokenized predicted NACE2025 code
        target_ids = self.tokenizer(parsed.nace2025).get("input_ids")

        nace2025_logprobs = self.extract_sequence_logprobs(output.outputs[0].logprobs, target_ids)

        score = torch.exp(nace2025_logprobs).mean().item()

        # We set the confidence score based on the logprobs
        parsed.confidence = score
        return parsed

    def extract_sequence_logprobs(self, logprobs: List[Dict[int, Any]], target_ids: List[int]) -> torch.Tensor:
        """
        Extracts logprobs for the exact target_ids sequence from the list of logprobs.

        Args:
            logprobs: List of dicts with {token_id: Logprob}.
            target_ids: The exact sequence of token IDs you want to find.

        Returns:
            Tensor of logprobs for the matched sequence, or empty tensor if not found.
        """

        # Convert the list of logprobs to a list of token IDs
        ids_sequence = [list(tok.keys())[0] if tok else None for tok in logprobs]

        sequence_length = len(target_ids)

        for i in range(len(ids_sequence) - sequence_length + 1):
            window_ids = ids_sequence[i : i + sequence_length]

            if window_ids == target_ids:
                # Exact match found, extract corresponding logprobs
                window_logprobs = []
                for j in range(sequence_length):
                    logprob_obj = list(logprobs[i + j].values())[0]
                    window_logprobs.append(logprob_obj.logprob)
                return torch.tensor(window_logprobs)

        # If no match found, return empty or fill with -inf
        return torch.full((sequence_length,), float("-inf"))

    def process_outputs(self, outputs: List[RequestOutput]) -> pd.DataFrame:
        """
        Process the outputs from the LLM and return a DataFrame.
        """
        results = pd.DataFrame.from_records(
            [self._process_output(output, self.response_format).model_dump() for output in outputs]
        )
        return super().postprocess_results(results)
