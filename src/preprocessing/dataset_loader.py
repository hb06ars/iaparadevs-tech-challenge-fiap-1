
import pandas as pd


class DatasetLoader:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

        self.columns = [
            "id",
            "diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave_points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave_points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst",
            "symmetry_worst",
            "fractal_dimension_worst"
        ]

    def load(self):
        self.df = pd.read_csv(
            self.file_path,
            header=None,
            names=self.columns
        )

        return self.df

    def get_dataframe(self):
        return self.df

    def show_preview(self):
        print(self.df.head())

    def show_info(self):
        self.df.info()

    def show_statistics(self):
        print(self.df.describe())

    def show_diagnosis_distribution(self):
        print(self.df["diagnosis"].value_counts())
