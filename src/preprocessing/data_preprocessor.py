from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

        self.target_column = "diagnosis"
        self.id_column = "id"

        self.X_train = None
        self.X_validation = None
        self.X_test = None

        self.y_train = None
        self.y_validation = None
        self.y_test = None

    def separate_features_and_target(self):

        print("\n=== SEPARANDO FEATURES E TARGET ===")

        self.df[self.target_column] = (
            self.df[self.target_column].map({
                "B": 0,
                "M": 1
            })
        )

        if self.df[self.target_column].isnull().any():
            raise ValueError("Diagnóstico desconhecido encontrado.")

        self.X = self.df.drop(
            columns=[self.id_column, self.target_column]
        )

        self.y = self.df[self.target_column]

        print(f"Quantidade de features: {self.X.shape[1]}")
        print(f"Quantidade de registros: {self.X.shape[0]}")

        print("\nDistribuição do target:")
        print(self.y.value_counts().sort_index())

    def split_data(self):

        print("\n=== DIVIDINDO OS DADOS ===")

        # Primeiro: 70% treinamento e 30% temporário
        self.X_train, X_temp, self.y_train, y_temp = (
            train_test_split(
                self.X,
                self.y,
                test_size=0.30,
                random_state=42,
                stratify=self.y
            )
        )

        # Segundo: divide os 30% temporários igualmente
        self.X_validation, self.X_test, \
            self.y_validation, self.y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )

        print(f"Treinamento: {len(self.X_train)} registros")
        print(f"Validação: {len(self.X_validation)} registros")
        print(f"Teste: {len(self.X_test)} registros")

    def show_class_distribution(self):

        print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")

        datasets = {
            "Treinamento": self.y_train,
            "Validação": self.y_validation,
            "Teste": self.y_test
        }

        for name, target in datasets.items():
            print(f"\n{name}:")

            distribution = target.value_counts(
                normalize=True
            ).sort_index() * 100

            print(distribution.round(2))

    def execute(self):

        print("\n====================================")
        print("INICIANDO PRÉ-PROCESSAMENTO")
        print("====================================")

        self.separate_features_and_target()
        self.split_data()
        self.validate_split()
        self.show_class_distribution()

        print("\nPRÉ-PROCESSAMENTO CONCLUÍDO")

        return {
            "X_train": self.X_train,
            "X_validation": self.X_validation,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_validation": self.y_validation,
            "y_test": self.y_test
        }

    def validate_split(self):

        print("\n=== VALIDANDO SEPARAÇÃO DOS DADOS ===")

        train_indices = set(self.X_train.index)
        validation_indices = set(self.X_validation.index)
        test_indices = set(self.X_test.index)

        assert train_indices.isdisjoint(validation_indices)
        assert train_indices.isdisjoint(test_indices)
        assert validation_indices.isdisjoint(test_indices)

        assert (
                len(train_indices)
                + len(validation_indices)
                + len(test_indices)
                == len(self.df)
        )

        print("Nenhuma sobreposição entre os conjuntos.")
        print("Todos os registros foram distribuídos corretamente.")
