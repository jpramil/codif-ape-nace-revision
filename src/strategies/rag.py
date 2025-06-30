from langchain_core.output_parsers import PydanticOutputParser

from constants.paths import URL_SIRENE4_AMBIGUOUS_RAG
from constants.vector_db import COLLECTION_NAME, RERANKER_MODEL
from llm.prompting import generate_prompts_from_data
from llm.response import RAGResponse
from vector_db.loading import get_retriever

from .base import EncodeStrategy


class RAGStrategy(EncodeStrategy):
    @property
    def parser(self):
        return PydanticOutputParser(pydantic_object=RAGResponse)

    def get_prompts(self, data):
        retriever = get_retriever(COLLECTION_NAME, RERANKER_MODEL)
        return generate_prompts_from_data(data, self.parser, retriever=retriever)

    @property
    def output_path(self):
        return URL_SIRENE4_AMBIGUOUS_RAG

    def postprocess_results(self, df):
        pass
