from pathlib import Path

import pandas as pd


class ModelComparison:

    def __init__(self, output_dir):
        """Define e cria a pasta na qual será salvo o arquivo de comparação."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare(self, model_results):
        """Ordena os modelos por recall, F1 e acurácia e salva a comparação em CSV."""
        dataframe = pd.DataFrame(model_results)

        dataframe = dataframe.sort_values(
            by=["recall", "f1_score", "accuracy"],
            ascending=False
        )

        print("\n=== COMPARAÇÃO DOS MODELOS ===")
        print(
            dataframe.to_string(
                index=False,
                formatters={
                    "accuracy": "{:.4f}".format,
                    "precision": "{:.4f}".format,
                    "recall": "{:.4f}".format,
                    "f1_score": "{:.4f}".format
                }
            )
        )

        output_file = self.output_dir / "comparacao_modelos.csv"

        dataframe.to_csv(
            output_file,
            index=False
        )

        print(f"\nComparação salva em: {output_file}")

        return dataframe
