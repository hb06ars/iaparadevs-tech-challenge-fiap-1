from pathlib import Path

import pandas as pd

from src.evaluation.model_evaluator import ModelEvaluator
from src.exploration.data_exploration import DataExploration
from src.models.logistic_regression import LogisticRegressionModel
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.dataset_loader import DatasetLoader


class Main:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]

        dataset_path = (
                self.project_root
                / "data"
                / "breast_cancer"
                / "breast_cancer.csv"
        )

        self.graphics_path = (
                self.project_root
                / "results"
                / "graphics"
        )

        self.dataset_loader = DatasetLoader(dataset_path)


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

        # TREINAMENTO DE REGRESSÃO LOGÍSTICA
        logistic_model = LogisticRegressionModel()

        logistic_model.train(
            datasets["X_train"],
            datasets["y_train"]
        )

        # Primeiras previsões utilizando a validação
        predictions = logistic_model.predict(
            datasets["X_validation"]
        )

        print("\n=== PRIMEIRAS PREVISÕES ===")
        print(predictions[:10])

        print("\n=== DISTRIBUIÇÃO DAS PREVISÕES ===")
        print(pd.Series(predictions).value_counts().sort_index())

        # AVALIAÇÃO DO MODELO
        evaluator = ModelEvaluator(
            output_dir=self.graphics_path
        )

        metrics = evaluator.evaluate(
            y_true=datasets["y_validation"],
            y_pred=predictions,
            model_name="Regressao Logistica"
        )

        print("\n=== RESULTADO DAS MÉTRICAS ===")
        print(metrics)


if __name__ == "__main__":
    application = Main()
    application.execute()
