import unicodedata
import pandas as pd


def normalize_value(value):
    return unicodedata.normalize("NFKC", value) if pd.notna(value) else None


def create_mapping(mapping_table: pd.DataFrame, explanatory_notes: pd.DataFrame) -> dict:
    """
    Processes the provided `mapping_table` and `explanatory_notes` DataFrames to create a
    hierarchical mapping between NAF 2008 and NAF 2025 codes, and enriches the mapping
    with additional explanatory notes.

    Parameters:
    ----------
    mapping_table : pd.DataFrame
        The DataFrame containing the correspondence between old and new NAF codes.
        Expected columns are: NAF 2008 codes, NAF 2025 codes, and their descriptions.

    explanatory_notes : pd.DataFrame
        The DataFrame containing explanatory notes for the NAF codes, including
        general notes, what the code comprises, and what it does not comprise.

    Returns:
    -------
    mapping : dict
        A nested dictionary mapping NAF 2008 codes (`naf08_niv5`) to:
        - "libelle" : The description of the NAF 2008 code.
        - "naf25" : A dictionary mapping NAF 2025 codes (`naf25_niv5`) to:
            - "libelle" : The description of the NAF 2025 code.
            - "notes" : General notes on the NAF 2025 code.
            - "comprend" : What the NAF 2025 code includes.
            - "comprend_pas" : What the NAF 2025 code does not include.
    """

    columns_mapping = {
        "NAFold-code\n(code niveau sous-classe de la nomenclature actuelle)": "naf08_niv5",
        "NACEold-code\n(niveau classe)": "naf08_niv4",
        "NAFold-intitulé\n(niveau sous-classe)": "lib_naf08_niv5",
        "NACEnew-code\n(niveau classe)": "naf25_niv4",
        "NAFnew-code\n(code niveau sous-classe de la nomenclature 2025, correspondance logique avec les NAFold-codes)": "naf25_niv5",
        "NAFnew-intitulé\n(niveau sous-classe)": "lib_naf25_niv5",
    }

    mapping_table = (
        mapping_table.iloc[:, [1, 3, 2, 10, 5, 11]]
        .rename(columns=columns_mapping)
        .assign(
            naf08_niv5=mapping_table.iloc[:, 1].str.replace(".", "", regex=False),
            naf08_niv4=mapping_table.iloc[:, 3].str.replace(".", "", regex=False),
            naf25_niv4=mapping_table.iloc[:, 5].str.replace(".", "", regex=False),
            naf25_niv5=mapping_table.iloc[:, 10].str.replace(".", "", regex=False),
        )
    )

    mapping = {
        code08: {
            "libelle": subset["lib_naf08_niv5"].iloc[0],
            "naf25": {
                row["naf25_niv5"]: {"libelle": row["lib_naf25_niv5"]}
                for _, row in subset.iterrows()
            },
        }
        for code08, subset in mapping_table.groupby("naf08_niv5")
    }

    explanatory_notes = (
        explanatory_notes.iloc[:, [6, 9, 10, 11, 12, 13, 7, 8]]
        .rename(
            columns={
                "Code.NACE.Rev.2.1": "naf08_niv4",
                "Titre": "lib_naf08_niv5",
                "Note.générale": "notes",
            }
        )
        .assign(naf08_niv4=explanatory_notes["Code.NACE.Rev.2.1"].str.replace(".", "", regex=False))
    )

    # Filling `mapping` with notes information
    for code08, code08_data in mapping.items():
        for code25, code25_data in code08_data["naf25"].items():
            # Attempt to find level 5 naf code in explanatory_notes
            row = explanatory_notes.loc[explanatory_notes["naf08_niv4"] == code25]

            # If doesn't exist, try to find level 4 naf code in explanatory_notes
            if row.empty:
                row = explanatory_notes.loc[explanatory_notes["naf08_niv4"] == code25[:-1]]
                if row.shape[0] != 1:
                    row = explanatory_notes.loc[
                        (explanatory_notes["naf08_niv4"] == code25[:-1])
                        & (explanatory_notes["indic.NAF"] == "1")
                    ]
                    if row.shape[0] != 1:
                        raise ValueError(f"Could not find notes for {code25}")

            # Extract relevant fields and normalize values
            notes = normalize_value(row["notes"].iloc[0])
            comprend = normalize_value(row["Comprend"].iloc[0])
            comprend_aussi = normalize_value(row["Comprend.aussi"].iloc[0])
            comprend_pas = normalize_value(row["Ne.comprend.pas"].iloc[0])

            # Combine 'comprend' and 'comprend_aussi' if they exist
            comprend_all = (
                "\n".join(filter(None, [comprend, comprend_aussi]))
                if comprend or comprend_aussi
                else None
            )

            # Update the `mapping` dictionary with the new information
            mapping[code08]["naf25"][code25].update(
                {
                    "notes": notes,
                    "comprend": comprend_all,
                    "comprend_pas": comprend_pas,
                }
            )

    return mapping
