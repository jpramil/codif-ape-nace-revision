import pandas as pd
import unicodedata

table_corres = pd.read_excel("table-correspondance-naf2025.xls", dtype=str)


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
mapping = {}
for code08 in naf08:
    subset = table_corres.loc[table_corres["naf08_niv5"] == code08]
    mapping[code08] = {}
    mapping[code08]["libelle"] = pd.unique(subset["lib_naf08_niv5"])[0]
    mapping[code08]["naf25"] = {
        row["naf25_niv5"]: {"libelle": row["lib_naf25_niv5"]} for _, row in subset.iterrows()
    }

notes_ex = pd.read_excel("notes-explicatives-naf2025.xlsx", dtype=str)
notes_ex["Code.NACE.Rev.2.1"] = notes_ex["Code.NACE.Rev.2.1"].str.replace(".", "", regex=False)


notes_ex = notes_ex.iloc[:, [6, 9, 10, 11, 12, 13]].rename(
    columns={
        "Code.NACE.Rev.2.1": "naf08_niv4",
        "Titre": "lib_naf08_niv5",
        "Note.générale": "notes",
    }
)


for code08 in mapping.keys():
    for code25 in mapping[code08]["naf25"].keys():
        row = notes_ex[notes_ex["naf08_niv4"] == code25]
        if row.empty:
            if notes_ex[notes_ex["naf08_niv4"] == code25[:-1]].shape[0] == 1:
                row = notes_ex[notes_ex["naf08_niv4"] == code25[:-1]]
            else:
                print(f"PROBLEM WITH {code25}")
                continue

        notes = (
            unicodedata.normalize("NFKC", row.notes.iloc[0])
            if pd.notna(row.notes.iloc[0])
            else None
        )
        comprend = (
            unicodedata.normalize("NFKC", row.Comprend.iloc[0])
            if pd.notna(row.Comprend.iloc[0])
            else None
        )
        comprend_aussi = (
            unicodedata.normalize("NFKC", row["Comprend.aussi"].iloc[0])
            if pd.notna(row["Comprend.aussi"].iloc[0])
            else None
        )
        comprend_all = (
            "\n".join(filter(None, [comprend, comprend_aussi]))
            if any([comprend, comprend_aussi])
            else None
        )
        comprend_pas = (
            unicodedata.normalize("NFKC", row["Ne.comprend.pas"].iloc[0])
            if pd.notna(row["Ne.comprend.pas"].iloc[0])
            else None
        )
        notes_expli = {
            "notes": notes,
            "comprend": comprend_all,
            "comprend_pas": comprend_pas,
        }
        mapping[code08]["naf25"][code25] = mapping[code08]["naf25"][code25] | notes_expli
