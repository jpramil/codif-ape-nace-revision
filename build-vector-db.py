import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

from src.constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_MAPPING_TABLE,
)
from src.mappings.mappings import get_mapping
from src.utils.cache_models import get_file_system

fs = get_file_system()

with fs.open(URL_MAPPING_TABLE) as f:
    table_corres = pd.read_excel(f, dtype=str)

with fs.open(URL_EXPLANATORY_NOTES) as f:
    notes_ex = pd.read_excel(f, dtype=str)

mapping = get_mapping(notes_ex, table_corres)

codes_naf2025 = []
tmp = []
for code08 in mapping:
    for code25 in code08.naf2025:
        if code25.code not in tmp:
            tmp.append(code25.code)
            codes_naf2025.append(code25)

df_naf2025 = pd.DataFrame([c.__dict__ for c in codes_naf2025])


df_naf2025.loc[:, "content"] = [
    "\n\n".join(
        filter(
            None,
            [
                f"# {row.code} : {row.label}",
                f"## Explications des activités incluses dans la sous-classe\n{row.notes}"
                if row.notes
                else None,
                f"## Liste d'exemples d'activités incluses dans la sous-classe\n{row.include}"
                if row.include
                else None,
                f"## Liste d'exemples d'activités non incluses dans la sous-classe\n{row.not_include}"
                if row.not_include
                else None,
            ],
        )
    )
    for row in df_naf2025.itertuples()
]

document_list = DataFrameLoader(df_naf2025, page_content_column="content").load()
