from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:

    def __init__(self):
        """Configura uma floresta de árvores de decisão para classificar os registros."""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            class_weight=None,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        """Ajusta as árvores com as medidas (X_train) e os diagnósticos conhecidos (y_train)."""
        print("\n=== TREINANDO RANDOM FOREST ===")

        self.model.fit(X_train, y_train)

        print("Treinamento concluído com sucesso!")

    def predict(self, X):
        """Prevê o diagnóstico de cada registro recebido, como benigno ou maligno."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Retorna a probabilidade estimada de cada diagnóstico para cada registro."""
        return self.model.predict_proba(X)

    def get_model(self):
        """Devolve o classificador interno para uso em outras etapas do projeto."""
        return self.model
