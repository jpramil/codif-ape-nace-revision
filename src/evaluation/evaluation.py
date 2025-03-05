from typing import Dict, List, Optional

import pandas as pd


def check_prompt_mapping(prompt, ground_truth_df: pd.DataFrame) -> Dict:
    """
    Determines if a proposed code exists in ground truth and finds its position.
    """
    gt_filtered = ground_truth_df.loc[ground_truth_df["liasse_numero"] == prompt.id]

    if gt_filtered.empty:
        return {"liasse_numero": prompt.id, "mapping_ok": False, "position": None}

    manual_code = gt_filtered["apet_manual"].values[0]

    mapping_ok = manual_code in prompt.proposed_codes
    position = prompt.proposed_codes.index(manual_code) if mapping_ok else None

    return {
        "liasse_numero": prompt.id,
        "mapping_ok": mapping_ok,
        "position": position,
    }


def get_prompt_mapping(prompts: List, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Processes prompts and returns a DataFrame with liasse_numero, mapping_ok, and position.
    """
    return pd.DataFrame([check_prompt_mapping(prompt, ground_truth) for prompt in prompts])


def calculate_accuracy(eval_df: pd.DataFrame, filter_col: Optional[str] = None) -> Dict:
    """
    Calculates accuracy at different levels.
    If `filter_col` is provided, only considers rows where `filter_col` is True.
    """
    filtered_df = eval_df if filter_col is None else eval_df[eval_df[filter_col]]

    return {
        f"accuracy_{filter_col or 'overall'}_lvl_{i}": round(
            (filtered_df["apet_manual"].str[:i] == filtered_df["nace2025"].str[:i]).mean() * 100, 2
        )
        for i in [5, 4, 3, 2, 1]
    }
