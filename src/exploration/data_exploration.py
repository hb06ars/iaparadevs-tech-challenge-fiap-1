from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


class DataExploration:

    def __init__(self, dataframe, output_dir):
        self.df = dataframe.copy()
        self.output_dir = Path(output_dir)

        # Cria o diretório de gráficos, caso não exista
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Identifica as variáveis que não são características preditoras
        self.target_column = "diagnosis"
        self.id_column = "id"

        self.feature_columns = [
            column for column in self.df.columns
            if column not in [self.target_column, self.id_column]
        ]

        sns.set_theme(style="whitegrid")

    def analyze_data_quality(self):
        print("\n=== QUALIDADE DOS DADOS ===")

        print("\nValores ausentes por coluna:")
        print(self.df.isnull().sum())

        print("\nTotal de valores ausentes:")
        print(self.df.isnull().sum().sum())

        print("\nQuantidade de registros duplicados:")
        print(self.df.duplicated().sum())

        print("\nQuantidade de IDs duplicados:")
        print(self.df[self.id_column].duplicated().sum())

        print("\nTipos de dados:")
        print(self.df.dtypes)

    def analyze_statistics(self):
        print("\n=== ANÁLISE ESTATÍSTICA ===")

        statistics = self.df[self.feature_columns].describe()

        print(statistics)

        statistics.to_csv(
            self.output_dir / "estatisticas_descritivas.csv"
        )

    def plot_diagnosis_distribution(self):
        print("\n=== GERANDO GRÁFICO DE DIAGNÓSTICOS ===")

        plt.figure(figsize=(8, 5))

        ax = sns.countplot(
            data=self.df,
            x=self.target_column,
            hue=self.target_column,
            order=["B", "M"],
            palette={"B": "#2E86AB", "M": "#D1495B"},
            legend=False
        )

        ax.set_title("Distribuição dos Diagnósticos")
        ax.set_xlabel("Diagnóstico")
        ax.set_ylabel("Quantidade")

        for container in ax.containers:
            ax.bar_label(container)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "distribuicao_diagnosticos.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def plot_feature_distributions(self):
        print("\n=== GERANDO HISTOGRAMAS DAS CARACTERÍSTICAS ===")

        number_of_features = len(self.feature_columns)

        fig, axes = plt.subplots(
            nrows=6,
            ncols=5,
            figsize=(20, 22)
        )

        axes = axes.flatten()

        for index, column in enumerate(self.feature_columns):
            sns.histplot(
                data=self.df,
                x=column,
                hue=self.target_column,
                kde=True,
                ax=axes[index],
                element="step"
            )

            axes[index].set_title(column, fontsize=9)
            axes[index].set_xlabel("")
            axes[index].set_ylabel("Frequência")

        # Remove espaços vazios, caso existam
        for index in range(number_of_features, len(axes)):
            fig.delaxes(axes[index])

        fig.suptitle(
            "Distribuição das Características por Diagnóstico",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "distribuicao_caracteristicas.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def plot_correlation_matrix(self):
        print("\n=== GERANDO MATRIZ DE CORRELAÇÃO ===")

        correlation = self.df[self.feature_columns].corr()

        plt.figure(figsize=(22, 18))

        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            annot=False,
            square=True
        )

        plt.title("Matriz de Correlação das Características")

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "matriz_correlacao.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        correlation.to_csv(
            self.output_dir / "matriz_correlacao.csv"
        )

    def execute(self):
        print("\n====================================")
        print("INICIANDO ANÁLISE EXPLORATÓRIA")
        print("====================================")

        self.analyze_data_quality()
        self.analyze_statistics()

        self.plot_diagnosis_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_matrix()

        print("\n====================================")
        print("ANÁLISE EXPLORATÓRIA CONCLUÍDA")
        print(f"Gráficos salvos em: {self.output_dir}")
        print("====================================")
