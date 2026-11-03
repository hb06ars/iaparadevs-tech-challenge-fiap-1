
from pathlib import Path

from src.preprocessing.dataset_loader import DatasetLoader
from src.exploration.data_exploration import DataExploration
from src.preprocessing.data_preprocessor import DataPreprocessor

class Main:

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]

        dataset_path = (
            project_root
            / "data"
            / "breast_cancer"
            / "breast_cancer.csv"
        )

        graphics_path = (
            project_root
            / "results"
            / "graphics"
        )

        self.dataset_loader = DatasetLoader(dataset_path)
        self.graphics_path = graphics_path

    def execute(self):
        print("\n=== CARREGANDO DATASET ===")
        df = self.dataset_loader.load()

        print("\n=== DIMENSÕES ===")
        print(df.shape)

        print("\n=== PRIMEIROS REGISTROS ===")
        self.dataset_loader.show_preview()

        print("\n=== INFORMAÇÕES ===")
        self.dataset_loader.show_info()

        print("\n=== DISTRIBUIÇÃO DOS DIAGNÓSTICOS ===")
        self.dataset_loader.show_diagnosis_distribution()

        exploration = DataExploration(
            dataframe=df,
            output_dir=self.graphics_path
        )

        exploration.execute()

        print("\n=== RESUMO DOS DATASETS ===")
        preprocessor = DataPreprocessor(dataframe=df)
        datasets = preprocessor.execute()
        for name, dataset in datasets.items():
            print(f"{name}: {dataset.shape}")


if __name__ == "__main__":
    application = Main()
    application.execute()
