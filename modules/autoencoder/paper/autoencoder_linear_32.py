import tensorflow as tf
from tensorflow.keras import layers


from modules import constants
from modules.FEN_converter import FENConverter
from modules.autoencoder.base_autoencoder import BaseAutoEncoder

from modules.vectorization.FEN import FEN

LATENT_DIM = 32
INPUT_SIZE = constants.VECTOR_LENGTH


class Autoencoder_Linear32(BaseAutoEncoder):

    def __init__(self):
        super(Autoencoder_Linear32, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(768,)),
                layers.Dense(LATENT_DIM, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [layers.Dense(INPUT_SIZE, activation="relu")]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def load_weights(self, filepath):
        self.build((None, INPUT_SIZE))
        super(Autoencoder_Linear32, self).load_weights(filepath)

    def vectorize_FEN(self, fen: FEN):
        return FENConverter.to_vector(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.vector_to_FEN(vector)
