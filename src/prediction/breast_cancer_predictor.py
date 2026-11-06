
import numpy as np
import pandas as pd

from pathlib import Path

from src.persistence.model_serializer import ModelSerializer


class BreastCancerPredictor:

    def __init__(self, model_path):
        """Carrega o modelo treinado e guarda a ordem das medidas que ele espera receber."""
        self.model_path = Path(model_path)
        self.serializer = ModelSerializer()
        self.model = self.serializer.load(self.model_path)

        # Recupera os nomes das características utilizadas no treinamento
        self.feature_names = list(self.model.feature_names_in_)

    def validate_input(self, dataframe):
        """Confere formato, colunas e valores; por exemplo, rejeita medidas ausentes ou infinitas."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Os dados devem ser informados em um pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError("Nenhum registro foi informado.")

        missing_features = [
            feature for feature in self.feature_names
            if feature not in dataframe.columns
        ]

        extra_features = [
            feature for feature in dataframe.columns
            if feature not in self.feature_names
        ]

        if missing_features:
            raise ValueError(
                f"Características ausentes: {missing_features}"
            )

        if extra_features:
            raise ValueError(
                f"Características não reconhecidas: {extra_features}"
            )

        # Mantém a mesma ordem das características do treinamento
        dataframe = dataframe[self.feature_names].copy()

        # Converte os valores para numéricos
        for column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise"
            )

        # Verifica valores ausentes
        if dataframe.isnull().any().any():
            raise ValueError(
                "Os dados possuem valores ausentes."
            )

        # Verifica valores infinitos
        if not np.isfinite(dataframe.to_numpy(dtype=float)).all():
            raise ValueError(
                "Os dados possuem valores infinitos."
            )

        return dataframe

    def predict(self, dataframe):
        """Classifica um ou mais registros e retorna diagnóstico e probabilidades de cada classe."""

        validated_data = self.validate_input(dataframe)

        predictions = self.model.predict(validated_data)
        probabilities = self.model.predict_proba(validated_data)

        results = []

        for prediction, probability in zip(
            predictions,
            probabilities
        ):
            results.append({
                "diagnosis": self._get_diagnosis(int(prediction)),
                "probability_benign": float(probability[0]),
                "probability_malignant": float(probability[1])
            })

        return pd.DataFrame(results)

    def predict_one(self, features):
        """Classifica um único registro em formato dicionário e devolve o resultado como dicionário."""

        if not isinstance(features, dict):
            raise TypeError(
                "O registro deve ser informado como um dicionário."
            )

        dataframe = pd.DataFrame([features])

        return self.predict(dataframe).iloc[0].to_dict()

    @staticmethod
    def _get_diagnosis(prediction):
        """Converte a classe numérica (0 ou 1) para o texto Benigno ou Maligno."""

        diagnosis_mapping = {
            0: "Benigno",
            1: "Maligno"
        }

        if prediction not in diagnosis_mapping:
            raise ValueError(
                f"Classe de diagnóstico desconhecida: {prediction}"
            )

        return diagnosis_mapping[prediction]
