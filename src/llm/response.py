from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from src.llm.prompting import PromptData


class LLMResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide
        classification code, False otherwise."""
    )

    nace08_valid: Optional[bool] = Field(
        description="""True if the NACE08 classification seems valid with the description of the activity, False otherwise.""",
        default=None,
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code Empty if codable=False.""",
        default=None,
    )


def process_response(response: str, prompt: PromptData, parser: PydanticOutputParser) -> dict:
    try:
        validated_response = parser.parse(response)
    except ValueError as parse_error:
        # Log an error and return an un-codable response if parsing fails.
        print(f"Error processing row with id {prompt.id}: {parse_error}")
        validated_response = LLMResponse(
            codable=False, nace08_valid=None, nace2025=None
        )

    if validated_response.nace2025 not in prompt.proposed_codes:
        # Log an error if the class code is invalid.
        print(
            f"Error processing row with id {prompt.id}: Code not in the ISCO list --> {validated_response.nace2025}"
        )
        validated_response.codable = False

    return {
            **validated_response.dict(),
            "id": prompt.id,
        }
