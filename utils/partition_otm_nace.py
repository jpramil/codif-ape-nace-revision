import os
import sys
from urllib.parse import urlencode

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import s3fs


def get_filesystem():
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    return fs


def extract_otm_cases(
    df: pd.DataFrame,
):
    subset = df.copy()
    subset["surface"] = subset["surface"].apply(reclassify_surface)
    return subset[["text_description", "type_", "nature", "surface", "event"]]


def save_to_s3(table: pa.Table, bucket: str, path: str):
    fs = get_filesystem()
    # To partition data
    pq.write_to_dataset(
        table,
        root_path=f"s3://{bucket}/{path}/",
        partition_cols=["date"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )


def main(data_file_path: str, dashboard_path: str, api_path: str):  # , date_to_log: str):
    # Define file system
    fs = get_filesystem()

    # Open Dataset
    data = (
        ds.dataset(
            f"{data_file_path}/test_data_NAF2008.parquet",
            # partitioning=["date"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        # .filter(
        #     (ds.field("date") == f"date={date_to_log}")
        # )
        .to_pandas()
    )

    # Harmonize dataset for the query
    table = format_query(data)
    results = query_batch_api(table, api_path, prob_min=0.0)
    table = add_prediction_columns(data, results)
    # Remove 'date=' prefix from the 'date' column to partition again
    table["date"] = table["date"].str.replace("date=", "")
    arrow_table = pa.Table.from_pandas(table)
    save_to_s3(arrow_table, "projet-ape", f"{dashboard_path}")


if __name__ == "__main__":
    data_file_path = str(sys.argv[1])
    dashboard_path = str(sys.argv[2])
    api_path = str(sys.argv[3])

    main(data_file_path, dashboard_path, api_path)