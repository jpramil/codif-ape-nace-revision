from langchain_core.output_parsers import PydanticOutputParser

from constants.paths import URL_SIRENE4_AMBIGUOUS_CAG
from llm.prompting import generate_prompts_from_data
from llm.response import CAGResponse

from .base import EncodeStrategy


class CAGStrategy(EncodeStrategy):
    @property
    def parser(self):
        return PydanticOutputParser(pydantic_object=CAGResponse)

    def get_prompts(self, data):
        _, mapping_ambiguous = data
        return generate_prompts_from_data(data[0], self.parser, mapping=mapping_ambiguous)

    @property
    def output_path(self):
        return URL_SIRENE4_AMBIGUOUS_CAG

    def postprocess_results(self, df):
        # Apply the base postprocessing first
        df = super().postprocess_results(df)
        # Then apply specific CAG postprocessing
        df["nace08_valid"] = df["nace08_valid"].fillna("undefined").astype(str)
        return df
