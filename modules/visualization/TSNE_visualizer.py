import numpy as np
from sklearn.manifold import TSNE
import os
from datetime import datetime

from modules.embedding_test.position_set import PositionSet
from modules.visualization.Point import Point
from modules.visualization.TNSEPoint import TSNEPoint
from modules.autoencoder.base_autoencoder import BaseAutoEncoder

HTML_TEMPLATE = "template.html"
HTML_PLACEHOLDER_FOR_JSON = "HTML_PLACEHOLDER_FOR_JSON"
HTML_PLACEHOLDER_FOR_INFO = "HTML_PLACEHOLDER_FOR_INFO"


class TSNEVisualizer:

    @staticmethod
    def visualize_tsne(
        autoencoder: BaseAutoEncoder,
        position_sets: list[PositionSet],
        visualization_file_name: str = None,
        num_points_to_visualize: int = None,
    ):

        points = TSNEVisualizer.encode_position_sets(
            autoencoder, position_sets, num_points_to_visualize
        )
        tsne_points = TSNEVisualizer.reduce_with_tsne(points)
        info = TSNEVisualizer.get_info(autoencoder, position_sets)
        TSNEVisualizer.create_HTML_visualization(
            visualization_file_name, info, tsne_points
        )

    ### ENCODE

    @staticmethod
    def encode_position_sets(
        autoencoder: BaseAutoEncoder,
        position_sets: list[PositionSet],
        num_points_to_visualize: int = None,
    ):
        points = []
        for position_set in position_sets:
            points.extend(
                TSNEVisualizer.encode_points(
                    autoencoder, position_set, num_points_to_visualize
                )
            )

        return points

    @staticmethod
    def encode_points(
        autoencoder: BaseAutoEncoder,
        position_set: PositionSet,
        num_points_to_visualize: int = None,
    ):

        predicted_points = []

        for fen in position_set.FEN_positions:
            predicted_tensor = autoencoder.encode_FEN_position(fen)
            p = Point(predicted_tensor, fen, position_set.color)
            predicted_points.append(p)

            if (
                num_points_to_visualize is not None
                and len(predicted_points) >= num_points_to_visualize
            ):
                break

        return np.array(predicted_points)

    ### TSNE

    @staticmethod
    def reduce_with_tsne(points: list[Point]) -> list[TSNEPoint]:

        tsne = TSNE(n_components=2, random_state=42)
        predictions = TSNEVisualizer.get_predictions_as_list(points)
        predictions = predictions.reshape((predictions.shape[0], -1))
        tsne_result = tsne.fit_transform(predictions)
        return TSNEVisualizer.create_tsne_points(points, tsne_result)

    @staticmethod
    def get_predictions_as_list(points: list[Point]):
        predictions = []
        for point in points:
            predictions.append(point.prediction)

        return np.array(predictions)

    """
        the order of points and tsne_result has to be the same as the TSNE was applied
        eg. tsne_result[0] has to be the the outcome of points[0] etc.
    """

    @staticmethod
    def create_tsne_points(points: list[Point], tsne_result: list) -> list[TSNEPoint]:
        tsne_points: list[TSNEPoint] = []
        for index, point in enumerate(points):
            tsne_points.append(
                TSNEPoint(
                    tsne_result[index, 0],
                    tsne_result[index, 1],
                    point.position,
                    point.color,
                )
            )

        return tsne_points

    ### VISUALIZATION

    @staticmethod
    def create_HTML_visualization(
        visualization_file_name, info: str, tsne_points: list[TSNEPoint]
    ) -> None:

        tsne_points_json = TSNEVisualizer.stringify_tsne_points(tsne_points)
        html_template_src = TSNEVisualizer.get_HTML_template_src()

        with open(html_template_src, "r", encoding="utf-8") as template_html:
            html = template_html.read().replace(
                HTML_PLACEHOLDER_FOR_JSON, tsne_points_json
            )
            html = html.replace(HTML_PLACEHOLDER_FOR_INFO, info)

        if visualization_file_name is None or visualization_file_name == "":
            visualization_file_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(visualization_file_name + ".html", "w") as output_html:
            output_html.write(html)

    @staticmethod
    def stringify_tsne_points(tsne_points):
        return str([obj.__dict__ for obj in tsne_points])

    @staticmethod
    def get_HTML_template_src():
        module_directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(module_directory, HTML_TEMPLATE)

    @staticmethod
    def get_info(autoencoder: BaseAutoEncoder, positions_sets: list[PositionSet]):

        position_sets_info = " </br> DATASETS: "
        for position_set in positions_sets:
            position_sets_info += (
                # str(position_set.name) + "(" + ": " + str(position_set.color) + " "
                f"{position_set.name}({position_set.color})"
            )
            position_sets_info += str(" || ")

        info = autoencoder.__class__.__name__
        info += " " + position_sets_info

        return info
