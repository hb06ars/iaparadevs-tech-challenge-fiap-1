from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


class DataExploration:

    def __init__(self, dataframe, output_dir):
        """Prepara uma cópia dos dados e identifica diagnóstico, identificador e medidas preditoras."""
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
        """Exibe verificações básicas, como dados ausentes, linhas duplicadas e tipos de coluna."""
        print("\n=== QUALIDADE DOS DADOS ===")
        valores_ausentes_por_coluna = self.df.isnull().sum()
        total_valores_ausentes = valores_ausentes_por_coluna.sum()
        quantidade_registros_duplicados = self.df.duplicated().sum()
        quantidade_ids_duplicados = self.df[self.id_column].duplicated().sum()

        if valores_ausentes_por_coluna.sum() > 0 or quantidade_registros_duplicados > 0 or quantidade_ids_duplicados > 0:
            print("ATENÇÃO: Foram encontrados problemas de qualidade nos dados!")
            print("\nValores ausentes por coluna:")
            print(valores_ausentes_por_coluna)

            print("\nTotal de valores ausentes:")
            print(total_valores_ausentes)

            print("\nQuantidade de registros duplicados:")
            print(quantidade_registros_duplicados)

            print("\nQuantidade de IDs duplicados:")
            print(quantidade_ids_duplicados)
        else:
            print("Nenhum problema de qualidade encontrado nos dados.")

    def analyze_statistics(self):
        """Calcula estatísticas descritivas das medidas e salva os valores em CSV."""
        print("\n=== ANÁLISE ESTATÍSTICA DE COLUNAS (MÉDIA, MEDIANA, DESVIO PADRÃO...) ===")

        statistics = self.df[self.feature_columns].describe()

        print(statistics)

        statistics.to_csv(
            self.output_dir / "estatisticas_descritivas.csv"
        )

    def plot_diagnosis_distribution(self):
        """Gera um gráfico com a quantidade de casos benignos e malignos."""
        print("\n=== GERANDO GRÁFICO DE DIAGNÓSTICOS (TARGET) - (MALIGNO X BENIGNO) ===")
        print("Arquivo gerado: distribuicao_diagnosticos.png")

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
        """Gera histogramas para comparar a distribuição de cada medida entre diagnósticos."""
        print("\n=== GERANDO HISTOGRAMAS DAS CARACTERÍSTICAS ===")
        print("Histogramas para cada coluna, mostrando a frequência entre os diagnósticos Maligno (M) e Benigno (B).")
        print("Arquivo gerado: distribuicao_caracteristicas.png")

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
        """Exibe e salva correlações entre medidas; valores próximos de 1 indicam relação positiva.
        Importante para identificar quais medidas estão mais relacionadas entre si e com o diagnóstico.
        """
        print("\n=== GERANDO MATRIZ DE CORRELAÇÃO ===")
        print("O quanto uma variável está relacionada a outra, o quanto um aumento no valor de uma variável "
              "resulta em um aumento ou diminuição no valor da outra.")
        print("Arquivo gerado: matriz_correlacao.png")

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
        """Executa as verificações, estatísticas e gráficos da análise exploratória."""
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
