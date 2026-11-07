from pathlib import Path

import matplotlib.pyplot as plt
import shap


class SHAPAnalyzer:

    def __init__(self, output_dir):
        """Define e cria a pasta para salvar os gráficos de explicação do modelo."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, model, X_validation):
        """Estima como cada medida influencia as previsões e salva gráficos SHAP explicativos."""
        print("\n=== INICIANDO ANÁLISE SHAP ===")
        print("Quais variáveis \"empurraram\" a previsão para Maligno ou para Benigno.")

        # Recupera o classificador Random Forest
        classifier = model.get_model()

        # Cria o explicador SHAP
        explainer = shap.TreeExplainer(classifier)

        # Calcula as contribuições das características
        shap_values = explainer(X_validation)

        # Seleciona a classe 1: diagnóstico maligno
        malignant_shap_values = shap_values[:, :, 1]

        # Gráfico de resumo: impacto das características
        plt.figure(figsize=(12, 8))

        shap.plots.beeswarm(
            malignant_shap_values,
            max_display=15,
            show=False
        )

        plt.title(
            "SHAP - Impacto das Características na "
            "Classificação Maligna"
        )

        plt.tight_layout()

        beeswarm_path = (
                self.output_dir / "shap_beeswarm.png"
        )

        plt.savefig(
            beeswarm_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # Gráfico de importância global SHAP
        plt.figure(figsize=(12, 8))

        shap.plots.bar(
            malignant_shap_values,
            max_display=15,
            show=False
        )

        plt.title(
            "SHAP - Importância Global das Características"
        )

        plt.tight_layout()

        bar_path = self.output_dir / "shap_importance.png"

        plt.savefig(
            bar_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"\nGráfico Beeswarm salvo em: {beeswarm_path}")
        print(f"Gráfico de importância salvo em: {bar_path}")

        return shap_values
