from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMModel:

    def __init__(self):
        """Configura o SVM e a padronização, que coloca as medidas em escalas comparáveis."""
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
        """Ensina o SVM a separar os diagnósticos a partir dos exemplos de treinamento."""
        print("\n=== TREINANDO SVM ===")

        self.model.fit(X_train, y_train)

        print("Treinamento concluído com sucesso!")

    def predict(self, X):
        """Classifica registros novos e retorna a classe prevista para cada um."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Informa que este SVM não foi configurado para calcular probabilidades."""
        raise NotImplementedError(
            "Probabilidades não estão habilitadas no SVM."
        )

    def get_model(self):
        """Devolve o modelo interno para avaliação ou salvamento."""
        return self.model
