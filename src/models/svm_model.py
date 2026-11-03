from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMModel:

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=42
            ))
        ])

    def train(self, X_train, y_train):
        print("\n=== TREINANDO SVM ===")

        self.model.fit(X_train, y_train)

        print("Treinamento concluído com sucesso!")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError(
            "Probabilidades não estão habilitadas no SVM."
        )

    def get_model(self):
        return self.model
