from src.preprocessing.dataset_loader import DatasetLoader


class Main:

    def __init__(self):
        self.dataset_loader = DatasetLoader(
            "../data/breast_cancer/wdbc.data"
        )

    def execute(self):

        print("\n=== CARREGANDO DATASET ===")

        df = self.dataset_loader.load()

        print("\n=== DIMENSÕES ===")
        print(df.shape)

        print("\n=== PRIMEIROS REGISTROS ===")
        self.dataset_loader.show_preview()

        print("\n=== INFORMAÇÕES ===")
        self.dataset_loader.show_info()

        print("\n=== ESTATÍSTICAS ===")
        self.dataset_loader.show_statistics()

        print("\n=== DISTRIBUIÇÃO DOS DIAGNÓSTICOS ===")
        self.dataset_loader.show_diagnosis_distribution()


if __name__ == "__main__":
    application = Main()
    application.execute()
