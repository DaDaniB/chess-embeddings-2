from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from modules.autoencoder.autoencoder_metrics import Autoencoder_Metrics
from modules.PGN_reader import PGNReader
from modules.vectorization.FEN import FEN
from modules import constants


class BaseAutoEncoder(tf.keras.models.Model, ABC):

    def __init__(self, *args, **kwargs):
        self.training_metrics = Autoencoder_Metrics()
        self.curr_num_trained_on = 0
        super(BaseAutoEncoder, self).__init__(*args, **kwargs)

    @abstractmethod
    def call(self, x):
        pass

    @abstractmethod
    def load_weights(self, file_path: str):
        super(BaseAutoEncoder, self).load_weights(file_path)

    @abstractmethod
    def vectorize_FEN():
        pass

    def vectorize_FENs(self, FENs: list[FEN]):
        vecs = []
        for FEN in FENs:
            vecs.append(self.vectorize_FEN(FEN))
        return np.array(vecs)

    @abstractmethod
    def vector_to_FEN(self, vector):
        pass

    def vectors_to_FENs(self, vectors):
        FENs = []
        for vector in vectors:
            FENs.append(self.vector_to_FEN(vector))
        return FENs

    def train(
        self, PGN_file: str, num_positions: int, chunk_size: int = None, epochs: int = 1
    ):
        if chunk_size is None:
            chunk_size = num_positions

        positions_vectorized = self.get_position_vectors_from_PGN(
            PGN_file, num_positions
        )
        self.train_on_vectors(positions_vectorized, chunk_size, epochs)

        savename = self.__class__.__name__ + "nPositions" + str(num_positions)
        self.training_metrics.save(savename)
        self.save(savename)

    def get_position_vectors_from_PGN(self, PGN_file: str, num_positions: int):
        FEN_positions = PGNReader.read_unique_positions_from_file(
            PGN_file, num_positions
        )

        return self.vectorize_FENs(FEN_positions)

    def train_on_vectors(self, positions_vectorized, chunk_size, epochs):
        if positions_vectorized is None or len(positions_vectorized) <= 0:
            return

        for i in range(epochs):
            print(f"epoch {i} of {epochs}")

            for j in range(0, len(positions_vectorized), chunk_size):
                position_vector_chunk = positions_vectorized[j : j + chunk_size]
                if position_vector_chunk is None or len(position_vector_chunk) <= 0:
                    return

                position_chunk_tensor = tf.constant(
                    position_vector_chunk, dtype=constants.TF_DATA_TYPE
                )

                history = self.fit(
                    position_chunk_tensor, position_chunk_tensor, epochs=1, shuffle=True
                )
                self.training_metrics.add_history(history, chunk_size)

            self.training_metrics.add_epoch_data()

    def save(self, savename: str):
        self.save_weights(constants.MODELS_SAVE_DIR + "/" + savename + ".h5")
