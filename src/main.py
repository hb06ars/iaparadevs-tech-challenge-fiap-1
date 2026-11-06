import matplotlib

matplotlib.use("Agg")

from src.pipelines.breast_cancer_pipeline import BreastCancerPipeline


class Main:

    def execute(self):
        """Inicia o pipeline principal, que executa as etapas do projeto em sequência apenas do Câncer de Mama."""
        pipeline = BreastCancerPipeline()
        pipeline.execute()


if __name__ == "__main__":
    application = Main()
    application.execute()
