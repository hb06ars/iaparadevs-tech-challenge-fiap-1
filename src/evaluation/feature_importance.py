from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class FeatureImportanceAnalyzer:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, model, feature_names, top_n=15):
        print("\n=== ANÁLISE DE FEATURE IMPORTANCE ===")

        # Recupera as importâncias calculadas pelo Random Forest
        importances = model.get_model().feature_importances_

        # Organiza as características da mais importante para a menos importante
        dataframe = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })

        dataframe = dataframe.sort_values(
            by="importance",
            ascending=False
        )

        print(f"\nTOP {top_n} CARACTERÍSTICAS:")
        print(dataframe.head(top_n).to_string(index=False))

        # Salva todas as importâncias em CSV
        csv_path = self.output_dir / "feature_importance.csv"
        dataframe.to_csv(csv_path, index=False)

        # Gera gráfico com as principais características
        top_features = dataframe.head(top_n).sort_values(
            by="importance",
            ascending=True
        )

        plt.figure(figsize=(10, 7))

        sns.barplot(
            data=top_features,
            x="importance",
            y="feature",
            hue="feature",
            palette="viridis",
            legend=False
        )

        plt.title(
            f"Top {top_n} - Importância das Características"
        )
        plt.xlabel("Importância")
        plt.ylabel("Característica")

        plt.tight_layout()

        image_path = self.output_dir / "feature_importance.png"

        plt.savefig(
            image_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"\nCSV salvo em: {csv_path}")
        print(f"Gráfico salvo em: {image_path}")

        return dataframe
