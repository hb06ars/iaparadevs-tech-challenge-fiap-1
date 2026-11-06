from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:

    def __init__(self):
        """Configura o modelo e a padronização dos dados antes da classificação."""
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ])

    def train(self, X_train, y_train):
        """Ensina o modelo usando exemplos conhecidos: X contém medidas e y, os diagnósticos."""
        print("\n=== TREINANDO REGRESSÃO LOGÍSTICA ===")

        self.model.fit(X_train, y_train)

        print("Treinamento concluído com sucesso!")

    def predict(self, X):
        """Prevê a classe de cada registro; por exemplo, 0 para benigno e 1 para maligno."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Retorna a confiança estimada para cada classe, em vez de somente a classe prevista."""
        return self.model.predict_proba(X)

    def get_model(self):
        """Devolve o objeto do scikit-learn para avaliação, explicação ou salvamento."""
        return self.model
