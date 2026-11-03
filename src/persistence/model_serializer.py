from pathlib import Path

import joblib


class ModelSerializer:

    def save(self, model, file_path):
        """
        Salva um modelo treinado em disco.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, path)

        print(f"\nModelo salvo com sucesso: {path}")

    def load(self, file_path):
        """
        Carrega um modelo previamente salvo.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {path}"
            )

        model = joblib.load(path)

        print(f"\nModelo carregado com sucesso: {path}")

        return model
