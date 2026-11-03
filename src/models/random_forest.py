from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            class_weight=None,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        print("\n=== TREINANDO RANDOM FOREST ===")

        self.model.fit(X_train, y_train)

        print("Treinamento concluído com sucesso!")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_model(self):
        return self.model
