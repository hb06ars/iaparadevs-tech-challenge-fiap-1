
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate


class CrossValidationEvaluator:

    def __init__(self, output_dir, number_of_folds=5):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.number_of_folds = number_of_folds

        self.scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1"
        }

    def evaluate(self, models, X_train, y_train):

        print("\n=== INICIANDO VALIDAÇÃO CRUZADA ===")

        cross_validation = StratifiedKFold(
            n_splits=self.number_of_folds,
            shuffle=True,
            random_state=42
        )

        results = []

        for model_name, model in models.items():

            print(f"\nAvaliando: {model_name}")

            scores = cross_validate(
                estimator=model,
                X=X_train,
                y=y_train,
                cv=cross_validation,
                scoring=self.scoring,
                return_train_score=False,
                n_jobs=1,
                error_score="raise"
            )

            result = {
                "model": model_name
            }

            for metric in self.scoring:

                values = scores[f"test_{metric}"]

                mean_value = values.mean()
                std_value = values.std()

                result[f"{metric}_mean"] = mean_value
                result[f"{metric}_std"] = std_value

                print(
                    f"{metric}: "
                    f"{mean_value:.4f} "
                    f"(+/- {std_value:.4f})"
                )

            results.append(result)

        dataframe = pd.DataFrame(results)

        output_file = (
            self.output_dir / "validacao_cruzada.csv"
        )

        dataframe.to_csv(output_file, index=False)

        print("\n=== RESULTADO DA VALIDAÇÃO CRUZADA ===")
        print(dataframe.to_string(index=False))

        print(f"\nResultados salvos em: {output_file}")

        return dataframe
