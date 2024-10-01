from typing import Optional

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide
        classification code, False otherwise."""
    )

    class_code: Optional[str] = Field(
        description="""NACE 2025 classification code Empty if codable=False.""",
        default=None,
    )
