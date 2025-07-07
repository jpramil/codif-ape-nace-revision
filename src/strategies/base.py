from abc import ABC, abstractmethod
from typing import Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class EncodeStrategy(ABC):
    """
    Abstract base class for your encoding strategies (RAG or CAG).
    """

    @abstractmethod
    def get_prompts(self, data: Any) -> List[Any]:
        """
        Each strategy defines how it builds prompts.
        """
        pass

    @property
    @abstractmethod
    def output_path(self) -> str:
        """
        Each strategy defines its output path.
        """
        pass

    @abstractmethod
    def postprocess_results(self, df):
        """
        Hook for extra tweaks on the final dataframe.
        """
        df["nace2025"] = df["nace2025"].str.replace(".", "", regex=False)
        return df

    def _format_activity_description(self, row: Any) -> str:
        """
        Format the activity description from the row data.
        """
        activity = row.get("libelle").lower() if row.get("libelle").isupper() else row.get("libelle")

        if row.get("activ_sec_agri_et"):
            activity += f"\nPrécisions sur l'activité agricole : {row.get('activ_sec_agri_et').lower()}"

        if row.get("activ_nat_lib_et"):
            activity += f"\nAutre nature d'activité : {row.get('activ_nat_lib_et').lower()}"

        return activity

    def save_results(self, df: pd.DataFrame, third: int):
        """
        Save the results to the specified output path.
        """
        output_path = self.output_path.format(third=f"{third}" if third else "", i="{i}")

        pq.write_to_dataset(
            pa.Table.from_pandas(df),
            root_path="/".join(output_path.split("/")[:-1]),
            partition_cols=["codable"],
            basename_template=output_path.split("/")[-1],
            existing_data_behavior="overwrite_or_ignore",
            filesystem=self.fs,
        )
        return output_path.format(i=0)
