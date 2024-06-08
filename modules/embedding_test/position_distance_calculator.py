import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import tensorflow as tf
from sklearn.metrics import pairwise_distances

from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.embedding_test.position_set import PositionSet


class PositionDistanceCalculator:

    @staticmethod
    def calculate_distances(
        autoencoder: BaseAutoEncoder, position_sets: list[PositionSet], output_file
    ):
        encoded_positions = PositionDistanceCalculator.encode_position_sets(
            autoencoder, position_sets
        )
        PositionDistanceCalculator.calculate_distances_pairwise(
            encoded_positions, output_file
        )

    @staticmethod
    def encode_position_sets(
        autoencoder: BaseAutoEncoder, position_sets: list[PositionSet]
    ):
        encoded_position_sets = []

        for position_set in position_sets:
            encoded_position_sets.append(
                (
                    position_set.name,
                    PositionDistanceCalculator.encode_positions(
                        autoencoder, position_set
                    ),
                )
            )

        return encoded_position_sets

    @staticmethod
    def encode_positions(autoencoder: BaseAutoEncoder, position_set: PositionSet):
        predicted_vectors = []
        if position_set.FEN_positions is None:
            raise ValueError(
                "Can only calculate distance from PositionSet with given FEN positions"
            )

        for FEN in position_set.FEN_positions:
            predicted_vector = autoencoder.encode_FEN_position(FEN)
            predicted_vectors.append(predicted_vector)

        return np.array(predicted_vectors)

    @staticmethod
    def calculate_distances_pairwise(encoded_position_sets, output_file):

        for i, position_set_a in enumerate(encoded_position_sets):
            for j, position_set_b in enumerate(encoded_position_sets):
                if j < i:
                    continue

                mean_distance_between_a_and_b = (
                    PositionDistanceCalculator.average_euclidean_distance(
                        position_set_a[1], position_set_b[1]
                    )
                )
                result_text = f"Mean distance between {position_set_a[0]} and {position_set_b[0]} is: {mean_distance_between_a_and_b}\n"
                print(result_text)
                output_file.write(result_text)

    @staticmethod
    def average_euclidean_distance(position_set_a, position_set_b):

        set_a_reshaped = position_set_a.reshape(position_set_a.shape[0], -1)
        set_b_reshaped = position_set_b.reshape(position_set_b.shape[0], -1)

        distances = pairwise_distances(set_a_reshaped, set_b_reshaped)
        avg_dist = np.mean(distances)
        return avg_dist

    @staticmethod
    def average_cosine_similarity():
        pass
