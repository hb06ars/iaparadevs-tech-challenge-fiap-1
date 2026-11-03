from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true, y_pred, model_name):
        print(f"\n=== AVALIAÇÃO: {model_name} ===")

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

        self.plot_confusion_matrix(
            y_true,
            y_pred,
            model_name
        )

        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
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

        plt.title(f"Matriz de Confusão - {model_name}")
        plt.xlabel("Previsão do modelo")
        plt.ylabel("Diagnóstico real")

        plt.tight_layout()

        file_name = (
                model_name.lower().replace(" ", "_")
                + "_matriz_confusao.png"
        )

        plt.savefig(
            self.output_dir / file_name,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Matriz de confusão salva em: "
            f"{self.output_dir / file_name}"
        )
