import pandas as pd
import unicodedata

table_corres = pd.read_excel("table-correspondance-naf2025.xls", dtype=str)
notes_ex = pd.read_excel("notes-explicatives-naf2025.xlsx", dtype=str)


table_corres = (
    table_corres.iloc[:, [1, 3, 2, 10, 5, 11]]
    .rename(
        columns={
            "NAFold-code\n(code niveau sous-classe de la nomenclature actuelle)": "naf08_niv5",
            "NACEold-code\n(niveau classe)": "naf08_niv4",
            "NAFold-intitulé\n(niveau sous-classe)": "lib_naf08_niv5",
            "NACEnew-code\n(niveau classe)": "naf25_niv4",
            "NAFnew-code\n(code niveau sous-classe de la nomenclature 2025, correspondance logique avec les NAFold-codes)": "naf25_niv5",
            "NAFnew-intitulé\n(niveau sous-classe)": "lib_naf25_niv5",
        }
    )
    .apply(lambda col: col.str.replace(".", "", regex=False) if col.dtype == "object" else col)
)

naf08 = list(pd.unique(table_corres["naf08_niv5"]))

mapping = {
    code08: {
        "libelle": pd.unique(subset["lib_naf08_niv5"])[0],
        "naf25": {
            row["naf25_niv5"]: {"libelle": row["lib_naf25_niv5"]} for _, row in subset.iterrows()
        },
    }
    for code08, subset in table_corres.groupby("naf08_niv5")
}

notes_ex["Code.NACE.Rev.2.1"] = notes_ex["Code.NACE.Rev.2.1"].str.replace(".", "", regex=False)
notes_ex = notes_ex.iloc[:, [6, 9, 10, 11, 12, 13, 7, 8]].rename(
    columns={
        "Code.NACE.Rev.2.1": "naf08_niv4",
        "Titre": "lib_naf08_niv5",
        "Note.générale": "notes",
    }
)


def normalize_value(value):
    return unicodedata.normalize("NFKC", value) if pd.notna(value) else None


for code08 in mapping.keys():
    for code25 in mapping[code08]["naf25"].keys():
        row = notes_ex.loc[notes_ex["naf08_niv4"] == code25]

        if row.empty:
            row = notes_ex.loc[notes_ex["naf08_niv4"] == code25[:-1]]
            if row.shape[0] != 1:
                row = notes_ex.loc[
                    (notes_ex["naf08_niv4"] == code25[:-1]) & (notes_ex["indic.NAF"] == "1")
                ]
                if row.shape[0] != 1:
                    raise ValueError(f"Could not find notes for {code25}")

        notes = normalize_value(row.notes.iloc[0])
        comprend = normalize_value(row.Comprend.iloc[0])
        comprend_aussi = normalize_value(row["Comprend.aussi"].iloc[0])
        comprend_pas = normalize_value(row["Ne.comprend.pas"].iloc[0])

        comprend_all = (
            "\n".join(filter(None, [comprend, comprend_aussi]))
            if any([comprend, comprend_aussi])
            else None
        )

        notes_expli = {
            "notes": notes,
            "comprend": comprend_all,
            "comprend_pas": comprend_pas,
        }
        mapping[code08]["naf25"][code25].update(notes_expli)
