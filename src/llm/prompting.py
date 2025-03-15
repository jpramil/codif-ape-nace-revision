from collections import namedtuple
from typing import Any, List, Optional, Tuple

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_qdrant import QdrantVectorStore

from src.constants.prompting import (
    CLASSIF_PROMPT_CAG,
    CLASSIF_PROMPT_RAG,
    SYS_PROMPT_CAG,
    SYS_PROMPT_RAG,
)
from src.llm.response import RAGResponse

# TODO create a class instead of a namedtuple
PromptData = namedtuple("PromptData", ["id", "proposed_codes", "prompt"])


def format_code25(codes: list, paragraphs=["include", "not_include", "notes"]):
    return "\n\n".join(
        [
            f"{nace2025.code}: {nace2025.label}\n{extract_info(nace2025, paragraphs=paragraphs)}"
            for nace2025 in codes
        ]
    )


def format_code08(codes: list):
    return "\n\n".join([f"{nace08.code}: {nace08.label}" for nace08 in codes])


def format_docs(docs: list):
    """
    Format the retrieved documents to be included in the prompt.

    Parameters:
    ----------
    docs : list
        A list of documents retrieved from the dataset.

    Returns:
    -------
    str
        A formatted string containing the document content.
    """
    return "\n\n".join([f"{doc[0].page_content}" for doc in docs])


def extract_info(nace2025, paragraphs: list[str]):
    info = [
        getattr(nace2025, paragraph)
        for paragraph in paragraphs
        if getattr(nace2025, paragraph) is not None
    ]
    return "\n\n".join(info) if info else ""


def build_activity_description(row) -> str:
    activity = row.libelle.lower() if row.libelle.isupper() else row.libelle

    if row.activ_sec_agri_et:
        activity += f"\nPrécisions sur l'activité agricole : {row.activ_sec_agri_et.lower()}"

    if row.activ_nat_lib_et:
        activity += f"\nAutre nature d'activité : {row.activ_nat_lib_et.lower()}"

    return activity


def create_specific_prompt_rag(activity: str, parser: Any, retriever: Any) -> str:
    retrieved_docs = retriever.similarity_search_with_relevance_scores(
        query=f"query : {activity}", k=5, score_threshold=0.5
    )

    prompt = CLASSIF_PROMPT_RAG.format(
        activity=activity,
        proposed_codes=format_docs(retrieved_docs),
        format_instructions=parser.get_format_instructions(),
    )
    proposed_codes = [c[0].metadata["code"] for c in retrieved_docs]

    return prompt, proposed_codes


def create_specific_prompt_cag(activity: str, parser: Any, row: Any, mapping: Any) -> str:
    nace08 = f"{row.apet_finale[:2]}.{row.apet_finale[2:]}"

    proposed_codes = next((m.naf2025 for m in mapping if m.code == nace08))

    prompt = CLASSIF_PROMPT_CAG.format(
        activity=activity,
        nace08=format_code08([next((m for m in mapping if m.code == nace08))]),
        proposed_codes=format_code25(
            proposed_codes, paragraphs=["include", "not_include", "notes"]
        ),
        format_instructions=parser.get_format_instructions(),
    )
    return prompt, [c.code for c in proposed_codes]


def generate_prompt(
    row: pd.Series,
    parser: PydanticOutputParser,
    retriever: Optional[QdrantVectorStore] = None,
    mapping: Optional[Any] = None,
) -> Tuple[int, List[str], str, str]:
    row_id = row.liasse_numero
    activity = build_activity_description(row)

    if parser.pydantic_object is RAGResponse:
        if retriever is None:
            raise ValueError("Retriever instance must be provided for RAGResponse.")
        prompt, proposed_codes = create_specific_prompt_rag(activity, parser, retriever)
        system_prompt = SYS_PROMPT_RAG
    else:
        if mapping is None:
            raise ValueError("Mapping data must be provided for CAGResponses.")
        prompt, proposed_codes = create_specific_prompt_cag(activity, parser, row, mapping)
        system_prompt = SYS_PROMPT_CAG

    return row_id, proposed_codes, prompt, system_prompt


def create_prompt_data_obj(
    row: pd.Series,
    parser: PydanticOutputParser,
    retriever: Optional[QdrantVectorStore] = None,
    mapping: Optional[Any] = None,
) -> PromptData:
    row_id, proposed_codes, prompt, system_prompt = generate_prompt(row, parser, retriever, mapping)

    return PromptData(
        id=row_id,
        proposed_codes=proposed_codes,
        prompt=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )


def load_prompts_from_file(url: str, fs) -> List[PromptData]:
    prompts_df = pd.read_parquet(url, filesystem=fs)
    return [
        PromptData(
            id=row_id,
            proposed_codes=proposed_codes.tolist(),
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        for row_id, proposed_codes, prompt, system_prompt in prompts_df.itertuples(index=False)
    ]


def generate_prompts_from_data(
    data: pd.DataFrame,
    parser: PydanticOutputParser,
    retriever: Optional[QdrantVectorStore] = None,
    mapping: Optional[Any] = None,
) -> List[PromptData]:
    return [
        create_prompt_data_obj(row, parser, retriever=retriever, mapping=mapping)
        for row in data.itertuples()
    ]
