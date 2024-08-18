from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error


from modules.autoencoder.autoencoder_metrics import Autoencoder_Metrics
from modules.PGN_reader import PGNReader
from modules.vectorization.FEN import FEN
from modules import constants
from modules.autoencoder.batch_history import BatchHistory


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
    def load_default_weights(self):
        pass

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

    @abstractmethod
    def initialize(self):
        pass

    def train(
        self,
        PGN_file: str = None,
        num_positions: int = 100000,
        epochs: int = 1,
        TXT_file: str = None,
    ):
        positions_vectorized = None
        if TXT_file is not None:
            positions_vectorized = self.get_position_vectors_from_TXT(
                TXT_file, num_positions
            )
        else:
            positions_vectorized = self.get_position_vectors_from_PGN(
                PGN_file, num_positions
            )

        self.train_on_vectors(positions_vectorized, epochs)
        self.save(savename=self.__class__.__name__ + "nPositions" + str(num_positions))

    def get_position_vectors_from_PGN(self, PGN_file: str, num_positions: int):
        FEN_positions = PGNReader.read_unique_positions_from_file(
            PGN_file, num_positions
        )
        return self.vectorize_FENs(FEN_positions)

    def get_position_vectors_from_TXT(self, TXT_file: str, num_positions: int = None):
        FEN_positions = PGNReader.read_positions_from_txt(TXT_file, num_positions)
        return self.vectorize_FENs(FEN_positions)

    def train_on_vectors(self, positions_vectorized, epochs):
        if positions_vectorized is None or len(positions_vectorized) <= 0:
            return

        positions_tensor = tf.constant(
            positions_vectorized, dtype=constants.TF_DATA_TYPE
        )

        batch_history = BatchHistory()
        history = self.fit(
            positions_tensor,
            positions_tensor,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            verbose=1,
            callbacks=[batch_history],
        )
        batch_history.plot_metrics(self.__class__.__name__)

    def save(self, savename: str):
        self.save_weights(constants.MODELS_SAVE_DIR + "/" + savename + ".h5")

    def encode_FEN_position_tensor(self, FEN: FEN):
        position_tensor = self.FEN_to_tensor(FEN)
        return self.encoder(position_tensor)

    def decode_tensor_toFen(self, tensor):
        decoded = self.decoder(tensor)
        return self.vector_to_FEN(decoded)

    def test_encode_decode(self, test_fen: str):
        position_tensor = self.FEN_to_tensor(test_fen)

        encoded = self.encoder(position_tensor).numpy()
        decoded = self.decoder(encoded)[0].numpy()

        decoded_rounded = np.floor(decoded + 0.5).astype(bool)
        predicted_fen = self.vector_to_FEN(decoded_rounded)

        print(test_fen)
        print(predicted_fen)

    def encode_FEN_position(self, FEN: FEN):
        return self.encode_FEN_position_tensor(FEN).numpy()

    def FEN_to_tensor(self, FEN: FEN):
        vector = self.vectorize_FEN(FEN)
        return tf.convert_to_tensor(np.array([vector]), dtype=constants.TF_DATA_TYPE)

    def test_on_txt_file(self, TXT_file: str, output_file):
        positions_vectorized = self.get_position_vectors_from_TXT(TXT_file)
        positions_tensor = tf.constant(
            positions_vectorized, dtype=constants.TF_DATA_TYPE
        )
        predicted_positions_tensor = self.predict(positions_tensor)
        mse = np.mean(np.square(positions_tensor - predicted_positions_tensor), axis=1)
        overall_mse = np.mean(mse)
        result_str = f"{TXT_file} Mean Squared Error: {overall_mse}"
        print(result_str)
        output_file.write(result_str + "\n")
