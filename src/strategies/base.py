from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.output_parsers import BaseOutputParser


class EncodeStrategy(ABC):
    """
    Abstract base class for your encoding strategies (RAG or CAG).
    """

    @property
    @abstractmethod
    def parser(self) -> BaseOutputParser:
        pass

    @abstractmethod
    def get_prompts(self, data: Any) -> List[Any]:
        """
        Each strategy defines how it builds prompts.
        """
        pass

    @property
    @abstractmethod
    def output_path(self) -> str:
        """
        Each strategy defines its output path.
        """
        pass

    @abstractmethod
    def postprocess_results(self, df):
        """
        Hook for extra tweaks on the final dataframe.
        """
        df["nace2025"] = df["nace2025"].str.replace(".", "", regex=False)
        return df
