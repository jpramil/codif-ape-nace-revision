import logging
import re
from typing import Any, Dict, List, Optional

import torch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class CAGResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide classification code, False otherwise."""
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code Empty if codable=False.""",
        default=None,
    )

    nace08_valid: Optional[bool] = Field(
        description="""True if the NACE08 classification seems valid with the description of the activity, False otherwise.""",
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
    prompt: Any,
    parser: PydanticOutputParser,
    logprobs: List = None,
) -> dict:
    try:
        validated_response = parser.parse(response)
    except ValueError as parse_error:
        # Log an error and return an un-codable response if parsing fails.
        logging.warning(f"Failed to parse response for id {prompt.id}: {parse_error}")
        if parser.pydantic_object is CAGResponse:
            validated_response = CAGResponse(codable=False, nace08_valid=None, nace2025=None)
        else:
            validated_response = RAGResponse(codable=False, nace2025=None)

    if validated_response.nace2025 not in prompt.proposed_codes:
        logging.warning(
            f"Invalid NACE2025 code '{validated_response.nace2025}' for row ID {prompt.id}"
        )
        validated_response.codable = False
        validated_response.nace2025 = None

    # Compute confidence if logprobs are provided
    confidence = None
    if logprobs and validated_response.nace2025:
        confidence = compute_confidence_score(logprobs)

    # Construct final response
    final_response = validated_response.model_dump()
    final_response["liasse_numero"] = prompt.id

    if confidence is not None:
        final_response["confidence"] = confidence

    return final_response


def extract_nace2025_logprobs(logprobs: List[Dict[int, Any]]):
    # Reconstruct the full string from tokens
    decoded_tokens = [list(tok.values())[0].decoded_token for tok in logprobs if tok]
    full_text = "".join(decoded_tokens)

    # Regex pattern to match the nace2025 value
    # It matches the exact structure: "nace2025": "<NN.NNL>"
    nace_pattern = r'"nace2025": "(\d{2}\.\d{2}[A-Za-z])"'
    match = re.search(nace_pattern, full_text)

    if not match:
        # If no match we return confidence of 0
        return torch.full((6,), float("-inf"))

    # Get character start and end indices of the nace code within the full text
    nace_start_char, nace_end_char = match.span(1)

    # Iterate again over tokens to map char indices to logprobs
    token_logprobs = []
    char_count = 0
    captured_chars = ""
    for tok in logprobs:
        if not tok:
            continue
        logprob_obj = list(tok.values())[0]
        token_str = logprob_obj.decoded_token
        token_len = len(token_str)

        token_start = char_count
        token_end = char_count + token_len

        # If the token overlaps with nace2025 substring, save its logprob
        if token_end > nace_start_char and token_start < nace_end_char:
            token_logprobs.append(logprob_obj.logprob)
            captured_chars += token_str

        char_count = token_end

        # Break as soon as we've captured all necessary chars
        if len(captured_chars) >= (nace_end_char - nace_start_char):
            break

    # Sanity check: Ensure captured chars match the nace pattern
    if not re.fullmatch(r"\d{2}\.\d{2}[A-Za-z]", captured_chars):
        logging.warning(
            f"Captured chars '{captured_chars}' do not match nace2025 pattern. Returning confidence 0."
        )
        return torch.full((6,), float("-inf"))

    return torch.tensor(token_logprobs)


def compute_confidence_score(logprobs: List[Dict[str, Any]]) -> float:
    """
    Computes confidence score based on log probabilities of generated tokens.

    Args:
        logprobs (List[Dict[str, Any]]): Log probabilities of generated tokens.

    Returns:
        float: The computed confidence score.
    """

    logprobs_tensor = extract_nace2025_logprobs(logprobs)

    return torch.exp(logprobs_tensor).mean().item()
