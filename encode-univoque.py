import pandas as pd
import duckdb
from src.utils.data import get_file_system
from src.mappings.mappings import create_mapping


def encore_univoque(
    url_source: str,
    url_out: str,
):
    fs = get_file_system()

    # Load excel files containing informations about mapping
    with fs.open("s3://projet-ape/NAF-revision/table-correspondance-naf2025.xls") as f:
        table_corres = pd.read_excel(f, dtype=str)

    with fs.open("s3://projet-ape/NAF-revision/notes-explicatives-naf2025.xlsx") as f:
        notes_ex = pd.read_excel(f, dtype=str)

    mapping = create_mapping(table_corres, notes_ex)

    # Select all univoque codes
    univoques = [naf08 for naf08 in mapping.keys() if len(mapping[naf08]["naf25"]) == 1]

    con = duckdb.connect(database=":memory:")

    # Construct the CASE statement from the dictionary mapping
    case_statement = "CASE "
    for key, value in mapping.items():
        case_statement += f"WHEN code_naf08 = '{key}' THEN '{[*value["naf25"]][0]}' "
    case_statement += "ELSE NULL END AS code_naf25"

    # SQL query with renamed column and new column using CASE for mapping
    query = f"""
        SELECT
            *,
            apet_finale AS code_naf08,
            {case_statement}
        FROM
            read_parquet('{url_source}')
        WHERE
            code_naf08 IN ('{"', '".join(univoques)}')
    """

    con.execute(
        f"""
        COPY
        ({query})
        TO '{url_out}'
        (FORMAT 'parquet')
    ;
    """
    )


if __name__ == "__main__":
    URL = "s3://projet-ape/extractions/20240812_sirene4.parquet"
    URL_OUT = "s3://projet-ape/NAF-revision/relabeled-data/20240812_sirene4_univoques.parquet"

    encore_univoque(URL, URL_OUT)
