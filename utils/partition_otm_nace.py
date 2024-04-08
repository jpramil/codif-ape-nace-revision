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


def save_to_s3(table: pa.Table, bucket: str, path: str):
    fs = get_filesystem()
    # To partition data
    pq.write_to_dataset(
        table,
        root_path=f"s3://{bucket}/{path}/",
        partition_cols=["apet_finale"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )


def main(data_file_path: str, partition_path: str):  # , date_to_log: str):
    # Define file system
    fs = get_filesystem()
    # List all the files in the prefix folder
    files = fs.ls(data_file_path)
    # Sort the files based on their modification time (last modified first)
    files_sorted = sorted(files, key=lambda x: fs.info(x)['LastModified'], reverse=True)
    # Get the last file in the sorted list
    last_file = files_sorted[0]
    # Open Dataset
    data = (
        ds.dataset(
            f"{last_file}",
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .to_pandas()
    )

    arrow_table = pa.Table.from_pandas(data)
    save_to_s3(arrow_table, "projet-ape", f"{partition_path}")


if __name__ == "__main__":
    data_file_path = str(sys.argv[1]) # "projet-ape/NAF-revision/extractions/one-to-many"
    partition_path = str(sys.argv[2]) # "NAF-revision/APE-partitions"
    main(data_file_path, partition_path)
