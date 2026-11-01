
from pathlib import Path
import pandas as pd


class DatasetLoader:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {self.file_path}"
            )

        self.df = pd.read_csv(self.file_path)

        # Remove colunas completamente vazias
        self.df = self.df.dropna(axis=1, how="all")

        return self.df

    def get_dataframe(self):
        return self.df

    def show_preview(self):
        print(self.df.head())

    def show_info(self):
        self.df.info()

    def show_statistics(self):
        print(self.df.describe())

    def show_diagnosis_distribution(self):
        print(self.df["diagnosis"].value_counts())
