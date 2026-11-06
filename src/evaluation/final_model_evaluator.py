from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class FinalModelEvaluator:

    def __init__(self, metrics_dir, graphics_dir):
        """Configura as pastas para salvar as métricas finais e a matriz de confusão."""
        self.metrics_dir = Path(metrics_dir)
        self.graphics_dir = Path(graphics_dir)

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.graphics_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true, y_pred):
        """Compara previsões com diagnósticos reais, calcula métricas e gera os resultados finais."""
        print("\n=== AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ===")

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true, y_pred, pos_label=1
        )

        recall = recall_score(
            y_true, y_pred, pos_label=1
        )

        f1 = f1_score(
            y_true, y_pred, pos_label=1
        )

        metrics = {
            "model": "Regressao Logistica",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")

        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Benigno", "Maligno"],
            zero_division=0
        ))

        self.save_metrics(metrics)
        self.plot_confusion_matrix(y_true, y_pred)

        return metrics

    def save_metrics(self, metrics):
        """Grava as métricas finais em CSV para consulta fora da execução do programa."""
        dataframe = pd.DataFrame([metrics])

        output_file = (
                self.metrics_dir / "avaliacao_final_teste.csv"
        )

        dataframe.to_csv(output_file, index=False)

        print(f"\nMétricas salvas em: {output_file}")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Gera um gráfico que mostra acertos e erros para cada diagnóstico."""
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        plt.figure(figsize=(7, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benigno", "Maligno"],
            yticklabels=["Benigno", "Maligno"]
        )

        plt.title("Matriz de Confusão - Avaliação Final")
        plt.xlabel("Previsão do modelo")
        plt.ylabel("Diagnóstico real")

        plt.tight_layout()

        output_file = (
                self.graphics_dir / "matriz_confusao_teste.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Matriz de confusão salva em: {output_file}")
