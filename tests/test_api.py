from io import BytesIO
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app


FEATURES = [f"feature_{index}" for index in range(30)]


class FakePredictor:
    def predict(self, dataframe):
        self.received_dataframe = dataframe.copy()
        missing = [feature for feature in FEATURES if feature not in dataframe.columns]
        extra = [feature for feature in dataframe.columns if feature not in FEATURES]
        if missing:
            raise ValueError(f"Características ausentes: {missing}")
        if extra:
            raise ValueError(f"Características não reconhecidas: {extra}")
        return pd.DataFrame([
            {
                "diagnosis": "Benigno",
                "probability_benign": 0.91,
                "probability_malignant": 0.09,
            }
            for _ in range(len(dataframe))
        ])


class PredictionApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.fake_predictor = FakePredictor()
        self.predictor_patch = patch(
            "src.api.get_predictor",
            return_value=self.fake_predictor,
        )
        self.predictor_patch.start()
        self.addCleanup(self.predictor_patch.stop)

    @staticmethod
    def make_csv(dataframe):
        return dataframe.to_csv(index=False, sep=",", decimal=".").encode("utf-8-sig")

    def test_upload_returns_classification_and_probabilities(self):
        dataframe = pd.DataFrame([{feature: 1.0 for feature in FEATURES}])
        dataframe.insert(0, "id", 123)
        csv_content = self.make_csv(dataframe)

        response = self.client.post(
            "/api/predict",
            files={
                "file": (
                    "medidas.csv",
                    csv_content,
                    "text/csv",
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["summary"], {"benign": 1, "malignant": 0})
        self.assertEqual(result["results"][0]["row"], "123")
        self.assertEqual(result["results"][0]["probability_benign"], 0.91)

    def test_upload_reads_dot_decimal_values(self):
        dataframe = pd.DataFrame([{feature: "1.25" for feature in FEATURES}])
        response = self.client.post(
            "/api/predict",
            files={"file": ("medidas.csv", self.make_csv(dataframe), "text/csv")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_predictor.received_dataframe.iloc[0, 0], 1.25)

    def test_upload_rejects_missing_features(self):
        csv_content = self.make_csv(pd.DataFrame({"feature_0": [1.0]}))

        response = self.client.post(
            "/api/predict",
            files={"file": ("incompleta.csv", csv_content)},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("Características ausentes", response.json()["detail"])

    def test_upload_rejects_non_csv_extension(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("medidas.xlsx", b"feature_0")},
        )

        self.assertEqual(response.status_code, 400)

    def test_upload_rejects_semicolon_delimited_csv(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("medidas.csv", b"feature_0;feature_1\n1.2;3.4\n")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("vírgula", response.json()["detail"])

    def test_home_and_example_template_are_available(self):
        page = self.client.get("/")
        template = self.client.get("/api/template")

        self.assertEqual(page.status_code, 200)
        self.assertIn("Demonstração experimental", page.text)
        self.assertIn("vírgula entre colunas, ponto nos decimais", page.text)
        self.assertEqual(template.status_code, 200)
        self.assertIn("text/csv", template.headers["content-type"])
        self.assertIn("modelo_medidas.csv", template.headers["content-disposition"])
        dataframe = pd.read_csv(BytesIO(template.content), sep=",", decimal=".")
        self.assertEqual(dataframe.shape, (4, 30))
        self.assertTrue(
            all(
                pd.api.types.is_numeric_dtype(dataframe[column])
                for column in dataframe
            )
        )


if __name__ == "__main__":
    unittest.main()
