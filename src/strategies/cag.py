import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
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
from constants.paths import URL_SIRENE4_AMBIGUOUS_CAG
from utils.data import fetch_mapping, get_file_system

from .base import EncodeStrategy

logger = logging.getLogger(__name__)


class CAGResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide classification code, False otherwise."""
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code. Empty if codable=False.""",
        default=None,
    )

    nace08_valid: Optional[bool] = Field(
        description="""True if the NACE08 classification seems valid with the description of the activity, False otherwise.""",
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


class CAGStrategy(EncodeStrategy):
    def __init__(
        self,
        generation_model: str = "gemma3:27b",
        reranker_model: str = None,
    ):
        self.fs = get_file_system()
        self.mapping = fetch_mapping()
        self.response_format = CAGResponse
        self.generation_model = generation_model
        self.prompt_template = Langfuse().get_prompt("cag-classifier", label="production")
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
        return f"{URL_SIRENE4_AMBIGUOUS_CAG}/{self.generation_model}/part-{{i}}-{{third}}--{date}.parquet"

    def postprocess_results(self, df):
        # Apply the base postprocessing first
        df = super().postprocess_results(df)
        # Then apply specific CAG postprocessing
        df["nace08_valid"] = df["nace08_valid"].fillna("undefined").astype(str)
        return df

    async def create_prompt(self, row: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        activity = self._format_activity_description(row)
        nace08 = f"{row.get('apet_finale')[:2]}.{row.get('apet_finale')[2:]}"
        nace_old, proposed_codes, list_codes = self._format_documents(nace08)

        return self.prompt_template.compile(
            activity=activity,
            nace_old=nace08,
            proposed_codes=proposed_codes,
            list_proposed_codes=list_codes,
        )

    def _format_documents(self, nace08) -> (str, str):
        nace2025_codes = next((m.naf2025 for m in self.mapping if m.code == nace08))
        nace08_code = next((m for m in self.mapping if m.code == nace08))

        nace_old = "\n\n".join([f"{c.code}: {c.label}" for c in [nace08_code]])
        list_codes = ", ".join([f"'{c.code}'" for c in nace2025_codes])
        proposed_codes = self.format_code(nace2025_codes)
        return nace_old, proposed_codes, list_codes

    def format_code(self, codes: list) -> str:
        return "\n\n".join([f"{c.code}: {c.label}\n{self.extract_info(c)}" for c in codes])

    def extract_info(self, code) -> str:
        info = [getattr(code, attr) for attr in ["include", "not_include", "notes"] if getattr(code, attr, None)]
        return "\n\n".join(info) if info else ""

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
        if parsed is None or parsed.nace2025 is None:
            return response_format(codable=False, nace2025=None, confidence=0.0)

        # We get the tokenized predicted NACE2025 code
        target_ids = self.tokenizer(parsed.nace2025).get("input_ids")

        nace2025_logprobs = self.extract_sequence_logprobs(output.outputs[0].logprobs, target_ids)

        score = torch.exp(nace2025_logprobs).mean().item()

        # We set the confidence score based on the logprobs
        parsed.confidence = score
        return parsed
