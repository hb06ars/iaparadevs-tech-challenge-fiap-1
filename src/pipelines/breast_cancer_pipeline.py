from pathlib import Path

import pandas as pd

from src.evaluation.cross_validation_evaluator import CrossValidationEvaluator
from src.evaluation.feature_importance import FeatureImportanceAnalyzer
from src.evaluation.final_model_evaluator import FinalModelEvaluator
from src.evaluation.model_comparison import ModelComparison
from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.shap_analyzer import SHAPAnalyzer
from src.exploration.data_exploration import DataExploration
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.svm_model import SVMModel
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.dataset_loader import DatasetLoader


class BreastCancerPipeline:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        self.dataset_path = (
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

        self.metrics_path = (
                self.project_root
                / "results"
                / "metrics"
        )

        self.dataset_loader = DatasetLoader(self.dataset_path)

        # Resultados disponíveis durante a execução
        self.dataframe = None
        self.datasets = None
        self.models = {}
        self.validation_metrics = []
        self.final_model = None
        self.final_metrics = None

    def execute(self):
        print("\n========================================")
        print(" INICIANDO PIPELINE DE CÂNCER DE MAMA")
        print("========================================")

        self.load_dataset()
        self.execute_exploration()
        self.prepare_datasets()
        self.train_and_evaluate_models()
        self.execute_feature_importance()
        self.execute_shap_analysis()
        self.execute_cross_validation()
        self.execute_final_evaluation()

        print("\n========================================")
        print(" PIPELINE FINALIZADO COM SUCESSO")
        print("========================================")

    def load_dataset(self):
        print("\n=== CARREGANDO DATASET ===")

        self.dataframe = self.dataset_loader.load()

        print("\n=== DIMENSÕES ===")
        print(self.dataframe.shape)

        print("\n=== PRIMEIROS REGISTROS ===")
        self.dataset_loader.show_preview()

        print("\n=== INFORMAÇÕES ===")
        self.dataset_loader.show_info()

        print("\n=== DISTRIBUIÇÃO DOS DIAGNÓSTICOS ===")
        self.dataset_loader.show_diagnosis_distribution()

    def execute_exploration(self):
        print("\n=== ANÁLISE EXPLORATÓRIA ===")

        exploration = DataExploration(
            dataframe=self.dataframe,
            output_dir=self.graphics_path
        )

        exploration.execute()

    def prepare_datasets(self):
        print("\n=== PRÉ-PROCESSAMENTO ===")

        preprocessor = DataPreprocessor(
            dataframe=self.dataframe
        )

        self.datasets = preprocessor.execute()

        print("\n=== RESUMO DOS DATASETS ===")

        for name, dataset in self.datasets.items():
            print(f"{name}: {dataset.shape}")

    def train_and_evaluate_models(self):
        print("\n=== TREINAMENTO E AVALIAÇÃO DOS MODELOS ===")

        evaluator = ModelEvaluator(
            output_dir=self.graphics_path
        )

        # Regressão Logística
        logistic_model = LogisticRegressionModel()

        logistic_model.train(
            self.datasets["X_train"],
            self.datasets["y_train"]
        )

        predictions = logistic_model.predict(
            self.datasets["X_validation"]
        )

        print("\n=== PRIMEIRAS PREVISÕES ===")
        print(predictions[:10])

        print("\n=== DISTRIBUIÇÃO DAS PREVISÕES ===")
        print(
            pd.Series(predictions)
            .value_counts()
            .sort_index()
        )

        logistic_metrics = evaluator.evaluate(
            y_true=self.datasets["y_validation"],
            y_pred=predictions,
            model_name="Regressao Logistica"
        )

        print("\n=== RESULTADO REGRESSÃO LOGÍSTICA ===")
        print(logistic_metrics)

        # Random Forest
        random_forest = RandomForestModel()

        random_forest.train(
            self.datasets["X_train"],
            self.datasets["y_train"]
        )

        rf_predictions = random_forest.predict(
            self.datasets["X_validation"]
        )

        rf_metrics = evaluator.evaluate(
            y_true=self.datasets["y_validation"],
            y_pred=rf_predictions,
            model_name="Random Forest"
        )

        print("\n=== RESULTADO RANDOM FOREST ===")
        print(rf_metrics)

        # SVM
        svm_model = SVMModel()

        svm_model.train(
            self.datasets["X_train"],
            self.datasets["y_train"]
        )

        svm_predictions = svm_model.predict(
            self.datasets["X_validation"]
        )

        svm_metrics = evaluator.evaluate(
            y_true=self.datasets["y_validation"],
            y_pred=svm_predictions,
            model_name="SVM"
        )

        print("\n=== RESULTADO SVM ===")
        print(svm_metrics)

        # Armazena os modelos para utilização nas próximas etapas
        self.models = {
            "Regressao Logistica": logistic_model,
            "Random Forest": random_forest,
            "SVM": svm_model
        }

        self.validation_metrics = [
            logistic_metrics,
            rf_metrics,
            svm_metrics
        ]

        self.compare_models()

    def compare_models(self):
        print("\n=== COMPARAÇÃO DOS MODELOS ===")

        comparison = ModelComparison(
            output_dir=self.metrics_path
        )

        comparison.compare(self.validation_metrics)

    def execute_feature_importance(self):
        print("\n=== IMPORTÂNCIA DAS CARACTERÍSTICAS ===")

        feature_analyzer = FeatureImportanceAnalyzer(
            output_dir=self.graphics_path
        )

        feature_analyzer.analyze(
            model=self.models["Random Forest"],
            feature_names=self.datasets["X_train"].columns,
            top_n=15
        )

    def execute_shap_analysis(self):
        print("\n=== ANÁLISE SHAP ===")

        shap_analyzer = SHAPAnalyzer(
            output_dir=self.graphics_path
        )

        shap_analyzer.analyze(
            model=self.models["Random Forest"],
            X_validation=self.datasets["X_validation"]
        )

    def execute_cross_validation(self):
        print("\n=== VALIDAÇÃO CRUZADA ===")

        cross_validator = CrossValidationEvaluator(
            output_dir=self.metrics_path,
            number_of_folds=5
        )

        models_for_validation = {
            name: model.get_model()
            for name, model in self.models.items()
        }

        cross_validator.evaluate(
            models=models_for_validation,
            X_train=self.datasets["X_train"],
            y_train=self.datasets["y_train"]
        )

    def execute_final_evaluation(self):
        print("\n=== PREPARANDO TREINAMENTO FINAL ===")

        # Junta treinamento e validação após a seleção do modelo
        X_final_train = pd.concat(
            [
                self.datasets["X_train"],
                self.datasets["X_validation"]
            ],
            ignore_index=True
        )

        y_final_train = pd.concat(
            [
                self.datasets["y_train"],
                self.datasets["y_validation"]
            ],
            ignore_index=True
        )

        print(f"Registros para treinamento: {len(X_final_train)}")
        print(
            "Registros reservados para teste: "
            f"{len(self.datasets['X_test'])}"
        )

        # Novo modelo, treinado com treinamento + validação
        self.final_model = LogisticRegressionModel()

        self.final_model.train(
            X_final_train,
            y_final_train
        )

        # O conjunto de teste permanece separado
        test_predictions = self.final_model.predict(
            self.datasets["X_test"]
        )

        final_evaluator = FinalModelEvaluator(
            metrics_dir=self.metrics_path,
            graphics_dir=self.graphics_path
        )

        self.final_metrics = final_evaluator.evaluate(
            y_true=self.datasets["y_test"],
            y_pred=test_predictions
        )

        print("\n=== RESULTADO FINAL ===")
        print(self.final_metrics)
