import pandas as pd
import pyarrow.parquet as pq

from src.constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_GROUND_TRUTH,
    URL_MAPPING_TABLE,
    URL_SIRENE4_MULTIVOCAL,
)
from src.mappings.mappings import get_mapping
from src.utils.data import get_file_system, merge_dataframes
from src.utils.strategies import (
    get_model_agreement_stats,
    select_labels_cascade,
    select_labels_voting,
    select_labels_weighted_voting,
)


def check_mapping(naf08, naf25):
    return naf25 in naf08_to_naf2025.get(naf08, set())


fs = get_file_system()

LLMS = [
    "hugging-quants--Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
    "neuralmagic--Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
    "Qwen--Qwen2.5-72B-Instruct-GPTQ-Int4",
    "mistralai/Ministral-8B-Instruct-2410",
]
WEIGHTS = [1, 1, 1, 1]
VAR_TO_KEEP = ["liasse_numero", "nace2025", "codable"]

df_dict = {
    llm_name.split("/")[0].split("--")[0]: pq.ParquetDataset(
        f"{URL_SIRENE4_MULTIVOCAL.replace("s3://", "")}/{"--".join(llm_name.split("/"))}",
        filesystem=fs,
    )
    .read()
    .to_pandas()
    for llm_name in LLMS
}

# TODO : TEMP
df_dict["Qwen"] = df_dict["Qwen"].loc[
    df_dict["Qwen"]["liasse_numero"].isin(df_dict["hugging-quants"]["liasse_numero"])
]
df_dict["hugging-quants"] = df_dict["hugging-quants"].loc[
    df_dict["hugging-quants"]["liasse_numero"].isin(df_dict["Qwen"]["liasse_numero"])
]
df_dict["neuralmagic"] = df_dict["neuralmagic"].loc[
    df_dict["neuralmagic"]["liasse_numero"].isin(df_dict["hugging-quants"]["liasse_numero"])
]

merged_df = merge_dataframes(
    df_dict,
    merge_on="liasse_numero",  # your merge column
    var_to_keep=VAR_TO_KEEP,  # your list of columns to keep
    columns_to_rename={"nace2025": "nace2025_{key}", "codable": "codable_{key}"},
)

model_columns = [f"nace2025_{model}" for model in df_dict.keys()]
weights = {f"nace2025_{model}": WEIGHTS[i] for i, model in enumerate(df_dict.keys())}

merged_df["cascade_label"] = select_labels_cascade(merged_df, model_columns)
merged_df["voting_label"] = select_labels_voting(merged_df, model_columns)
merged_df["weighted_voting_label"] = select_labels_weighted_voting(
    merged_df, model_columns, weights
)


ground_truth = (
    pq.ParquetDataset(URL_GROUND_TRUTH.replace("s3://", ""), filesystem=fs).read().to_pandas()
)
# TODO: TEMP DUPLICATED
ground_truth = ground_truth.loc[~ground_truth.duplicated(subset="liasse_numero")]

with fs.open(URL_MAPPING_TABLE) as f:
    table_corres = pd.read_excel(f, dtype=str)

with fs.open(URL_EXPLANATORY_NOTES) as f:
    notes_ex = pd.read_excel(f, dtype=str)

mapping = get_mapping(notes_ex, table_corres)


naf08_to_naf2025 = {m.code: [c.code for c in m.naf2025] for m in mapping}
ground_truth["mapping_ok"] = [
    check_mapping(naf08, naf25)
    for naf08, naf25 in zip(ground_truth["NAF2008_code"], ground_truth["apet_manual"])
]


results_df = merged_df.merge(
    ground_truth.loc[:, ["liasse_numero", "NAF2008_code", "apet_manual", "mapping_ok"]],
    on="liasse_numero",
)

accuracies_raw = {
    f"accuracy_{model.replace("nace2025_", "")}_lvl_{i}": round(
        (results_df["apet_manual"].str[:i] == results_df[f"{model}"].str[:i]).mean() * 100,
        2,
    )
    for i in [5, 4, 3, 2, 1]
    for model in [f"nace2025_{x}" for x in df_dict.keys()]
    + ["cascade_label", "voting_label", "weighted_voting_label"]
}

accuracies_codable = {
    f"accuracy_{model}_lvl_{i}": round(
        (
            results_df[results_df[f"codable_{model}"] == "true"]["apet_manual"].str[:i]
            == results_df[results_df[f"codable_{model}"] == "true"][f"nace2025_{model}"].str[:i]
        ).mean()
        * 100,
        2,
    )
    for i in [5, 4, 3, 2, 1]
    for model in df_dict.keys()
}

accuracies_raw_llm = {
    f"accuracy_{model.replace("nace2025_", "")}_lvl_{i}": round(
        (
            results_df[results_df["mapping_ok"]]["apet_manual"].str[:i]
            == results_df[results_df["mapping_ok"]][f"{model}"].str[:i]
        ).mean()
        * 100,
        2,
    )
    for i in [5, 4, 3, 2, 1]
    for model in [f"nace2025_{x}" for x in df_dict.keys()]
    + ["cascade_label", "voting_label", "weighted_voting_label"]
}

stats = get_model_agreement_stats(results_df, model_columns)

print(f"Raw accuracies : {accuracies_raw}\n\n")
print(f"Codable accuracies : {accuracies_codable}\n\n")
print(f"Raw LLM accuracies : {accuracies_raw_llm}\n\n")
print(f"---------------------------------\nSTATISTIQUES\n {stats}\n\n")
