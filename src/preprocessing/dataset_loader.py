from pathlib import Path

import pandas as pd


class DatasetLoader:

    def __init__(self, file_path):
        """Armazena o caminho do arquivo CSV que contém os dados do projeto."""
        self.file_path = Path(file_path)
        self.df = None

    def load(self):
        """Lê o CSV em uma tabela pandas e remove colunas completamente vazias."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {self.file_path}"
            )

        self.df = pd.read_csv(self.file_path)

        # Remove colunas completamente vazias
        self.df = self.df.dropna(axis=1, how="all")

        return self.df

    def get_dataframe(self):
        """Devolve a tabela carregada, permitindo que outras etapas reutilizem os dados."""
        return self.df

    def show_preview(self):
        """Mostra as primeiras linhas para conferir rapidamente o conteúdo do arquivo."""
        print(self.df.head())

    def show_info(self):
        """Mostra tipos de dados e quantidade de valores preenchidos em cada coluna."""
        self.df.info()

    def show_statistics(self):
        """Exibe estatísticas básicas, como média, desvio e valores mínimo e máximo."""
        print(self.df.describe())

    def show_diagnosis_distribution(self):
        """Conta quantos registros existem por diagnóstico, por exemplo, B e M."""
        print(self.df["diagnosis"].value_counts())
