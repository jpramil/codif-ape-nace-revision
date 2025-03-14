import re
from collections import defaultdict, namedtuple

from src.constants.prompting import (
    CLASSIF_PROMPT_CAG,
    CLASSIF_PROMPT_RAG,
    SYS_PROMPT_CAG,
    SYS_PROMPT_RAG,
)

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
        re.sub(
            r"\d{2}\.\d{2}[A-Z]?|\d{2}\.\d{1}[A-Z]?|, voir groupe|, voir",
            "",
            getattr(nace2025, paragraph),
        )
        for paragraph in paragraphs
        if getattr(nace2025, paragraph) is not None
    ]
    return "\n\n".join(info) if info else ""


def generate_prompt_cag(row, mapping):
    nace08 = row.apet_finale
    activity = row.libelle.lower() if row.libelle.isupper() else row.libelle
    row_id = row.liasse_numero

    specs_agriculture = row.activ_sec_agri_et
    specs_nature = row.activ_nat_lib_et

    if specs_agriculture is not None:
        activity += f"\nPrécisions sur l'activité agricole : {specs_agriculture.lower()}"
    if specs_nature is not None:
        activity += f"\nAutre nature d'activité : {specs_nature.lower()}"

    proposed_codes = next((m.naf2025 for m in mapping if m.code == nace08))

    prompt = CLASSIF_PROMPT_CAG.format(
        **{
            "activity": activity,
            "nace08": format_code08([next((m for m in mapping if m.code == nace08))]),
            "proposed_codes": format_code25(
                proposed_codes, paragraphs=["include", "not_include", "notes"]
            ),
        }
    )
    return PromptData(
        id=row_id,
        proposed_codes=[c.code for c in proposed_codes],
        prompt=[
            {"role": "system", "content": SYS_PROMPT_CAG},
            {"role": "user", "content": prompt},
        ],
    )


def generate_prompt_rag(row, retriever):
    """
    Generate a prompt for the LLM model.

    Parameters:
    ----------
    row : pd.Series
        A row of the dataset.
    retriever : QdrantVectorStore
        A retriever object to retrieve relevant documents.

    Returns:
    -------
    PromptData
        A dataclass containing the prompt and the relevant information.
    """

    activity = row.libelle.lower() if row.libelle.isupper() else row.libelle
    row_id = row.liasse_numero

    specs_agriculture = row.activ_sec_agri_et
    specs_nature = row.activ_nat_lib_et

    if specs_agriculture is not None:
        activity += f"\nPrécisions sur l'activité agricole : {specs_agriculture.lower()}"
    if specs_nature is not None:
        activity += f"\nAutre nature d'activité : {specs_nature.lower()}"

    retrieved_docs = retriever.similarity_search_with_relevance_scores(
        query=f"query : {activity}", k=5, score_threshold=0.5
    )

    # retrieved_docs = retriever.similarity_search(
    #             query=f"query : {activity}", k=5,
    #         )

    prompt = CLASSIF_PROMPT_RAG.format(
        **{"activity": activity, "proposed_codes": format_docs(retrieved_docs)}
    )
    return PromptData(
        id=row_id,
        proposed_codes=[c[0].metadata["code"] for c in retrieved_docs],
        prompt=[
            {"role": "system", "content": SYS_PROMPT_RAG},
            {"role": "user", "content": prompt},
        ],
    )


def apply_template(messages, template):
    # Define an inner function to handle the core logic for one list of messages
    def format_single_prompt(message_list):
        prompt_data = {f"{message['role']}_prompt": message["content"] for message in message_list}
        return template.format_map(defaultdict(str, prompt_data))

    # Check if input is a list of lists
    if isinstance(messages[0], list):
        # It's a list of lists, return a list of formatted prompts
        return [format_single_prompt(message_list) for message_list in messages]
    else:
        # It's a single list of messages, return one formatted prompt
        return format_single_prompt(messages)
