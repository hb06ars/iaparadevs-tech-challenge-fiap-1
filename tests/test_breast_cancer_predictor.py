
from pathlib import Path

from src.prediction.breast_cancer_predictor import BreastCancerPredictor
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.dataset_loader import DatasetLoader


class BreastCancerPredictorTest:

    def execute(self):
        """Exercita a previsão com uma linha do teste e também com um registro em formato dict."""
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

        # Carrega e prepara o dataset
        loader = DatasetLoader(dataset_path)
        dataframe = loader.load()

        preprocessor = DataPreprocessor(dataframe)
        datasets = preprocessor.execute()

        # Inicializa o preditor
        predictor = BreastCancerPredictor(model_path)

        # Seleciona somente um registro do conjunto de teste
        patient_data = datasets["X_test"].iloc[[0]]

        print("\n=== DADOS DE ENTRADA ===")
        print(patient_data)

        # Realiza a previsão
        result = predictor.predict(patient_data)

        print("\n=== RESULTADO DA PREDIÇÃO ===")
        print(result.to_string(index=False))

        # Também testa a entrada no formato dict
        patient_dict = patient_data.iloc[0].to_dict()

        single_result = predictor.predict_one(patient_dict)

        print("\n=== RESULTADO PREDICT_ONE ===")
        print(single_result)


if __name__ == "__main__":
    test = BreastCancerPredictorTest()
    test.execute()
