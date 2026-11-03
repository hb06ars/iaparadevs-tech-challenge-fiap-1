from pathlib import Path

from src.persistence.model_serializer import ModelSerializer
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.dataset_loader import DatasetLoader


class ModelLoadingTest:

    def execute(self):
        project_root = Path(__file__).resolve().parents[1]

        dataset_path = (
                project_root
                / "data"
                / "breast_cancer"
                / "breast_cancer.csv"
        )

        model_path = (
                project_root
                / "models"
                / "breast_cancer_logistic_regression.joblib"
        )

        # Carrega o dataset
        loader = DatasetLoader(dataset_path)
        dataframe = loader.load()

        # Recupera os conjuntos de dados
        preprocessor = DataPreprocessor(dataframe)
        datasets = preprocessor.execute()

        # Carrega o modelo salvo
        serializer = ModelSerializer()
        model = serializer.load(model_path)

        # Realiza previsões no conjunto de teste
        predictions = model.predict(datasets["X_test"])

        print("\n=== TESTE DE CARREGAMENTO ===")
        print(f"Previsões realizadas: {len(predictions)}")
        print("Primeiras previsões:")
        print(predictions[:10])

        print("\nModelo carregado e utilizado com sucesso!")


if __name__ == "__main__":
    test = ModelLoadingTest()
    test.execute()
