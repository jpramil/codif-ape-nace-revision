import pandas as pd


def merge_dataframes(df_dict: dict, merge_on, var_to_keep, columns_to_rename=None, how="inner"):
    """
    Merge a dictionary of pandas DataFrames.

    Parameters:
    -----------
    df_dict : dict
        Dictionary of pandas DataFrames to merge with their names
    merge_on : str or list
        Column(s) to merge on
    var_to_keep : list
        List of columns to keep from each DataFrame
    columns_to_rename : dict, optional
        Dictionary specifying which columns to rename with suffix for each DataFrame
        Example: {"nace2025": "nace2025_{key}", "codable": "codable_{key}"}
    how : str, default 'inner'
        Type of merge to be performed: 'left', 'right', 'outer', 'inner'

    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame
    """
    if not df_dict:
        raise ValueError("DataFrame dictionary is empty")

    # Create a copy of the dictionary to avoid modifying the original
    processed_dfs = {}

    # Process each DataFrame: select columns and rename as needed
    for key, df in df_dict.items():
        # Select columns to keep
        temp_df = df[var_to_keep].copy()

        # Rename columns if specified
        if columns_to_rename:
            rename_map = {
                col: pattern.format(key=key) for col, pattern in columns_to_rename.items()
            }
            temp_df.rename(columns=rename_map, inplace=True)

        processed_dfs[key] = temp_df

    # Start with the first DataFrame
    first_key = next(iter(processed_dfs))
    result = processed_dfs[first_key]

    # Merge with remaining DataFrames
    for key in list(processed_dfs.keys())[1:]:
        result = pd.merge(
            result,
            processed_dfs[key],
            on=merge_on,
            how=how,
        )

    return result
