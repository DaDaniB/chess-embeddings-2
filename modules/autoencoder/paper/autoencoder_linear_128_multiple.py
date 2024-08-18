import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from modules.autoencoder.weighted_mse import WeightedMSE
from modules import constants
from modules.FEN_converter import FENConverter
from modules.autoencoder.base_autoencoder import BaseAutoEncoder

from modules.vectorization.FEN import FEN

LATENT_DIM = 128
INPUT_SIZE = constants.VECTOR_LENGTH


class Autoencoder_Linear128_multiple(BaseAutoEncoder):

    def __init__(self):
        super(Autoencoder_Linear128_multiple, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(INPUT_SIZE,)),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(32),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(INPUT_SIZE, activation="relu"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def initialize(self):
        optimizer = Adam(learning_rate=1e-3)
        loss = WeightedMSE(weight_for_1=4.0)
        self.compile(optimizer=optimizer, loss=loss)

    def load_weights(self, filepath):
        self.build((None, INPUT_SIZE))
        super(Autoencoder_Linear128_multiple, self).load_weights(filepath)

    def load_default_weights(self):
        self.load_weights("./models/Autoencoder_Linear128_multiplenPositions3000000.h5")

    def vectorize_FEN(self, fen: FEN):
        return FENConverter.to_vector(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.vector_to_FEN(vector)
