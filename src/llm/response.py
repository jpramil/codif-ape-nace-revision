import logging
from typing import Any, Dict, List, Optional

import torch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.llm.prompting import PromptData


class LLMResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide classification code, False otherwise."""
    )

    nace08_valid: Optional[bool] = Field(
        description="""True if the NACE08 classification seems valid with the description of the activity, False otherwise.""",
        default=None,
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code Empty if codable=False.""",
        default=None,
    )


class RAGResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide classification code, False otherwise."""
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code Empty if codable=False.""",
        default=None,
    )


def process_response(
    response: str,
    prompt: PromptData,
    parser: PydanticOutputParser,
    logprobs: List = None,
    tokenizer=None,
) -> dict:
    try:
        validated_response = parser.parse(response)
    except ValueError as parse_error:
        # Log an error and return an un-codable response if parsing fails.
        logging.warning(f"Failed to parse response for id {prompt.id}: {parse_error}")
        validated_response = LLMResponse(codable=False, nace08_valid=None, nace2025=None)

    if validated_response.nace2025 not in prompt.proposed_codes:
        logging.warning(
            f"Invalid NACE2025 code '{validated_response.nace2025}' for row ID {prompt.id}"
        )
        validated_response.codable = False
        validated_response.nace2025 = None

    # Compute confidence if logprobs are provided
    confidence = None
    if logprobs and tokenizer and validated_response.nace2025:
        generated_tokens = tokenizer.encode(validated_response.nace2025)
        confidence = compute_confidence_score(logprobs, generated_tokens)

    # Construct final response
    final_response = validated_response.model_dump()
    final_response["liasse_numero"] = prompt.id

    if confidence is not None:
        final_response["confidence"] = confidence

    return final_response


def compute_confidence_score(logprobs: List[Dict[str, Any]], generated_tokens: List[int]) -> float:
    """
    Computes confidence score based on log probabilities of generated tokens.

    Args:
        logprobs (List[Dict[str, Any]]): Log probabilities of generated tokens.
        generated_tokens (List[int]): List of token IDs corresponding to the generated text.

    Returns:
        float: The computed confidence score.
    """
    if len(logprobs) < len(generated_tokens) - 1:
        logging.error("Logprobs array is shorter than the generated tokens. Check slicing.")
        return 0.0  # Return neutral confidence if issue occurs

    logprobs_tensor = torch.tensor(
        [
            logprobs[i][token].logprob if token in logprobs[i] else -100
            for i, token in enumerate(generated_tokens[1:])
        ]
    )

    return torch.exp(logprobs_tensor).mean().item()
