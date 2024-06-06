import tensorflow as tf
from tensorflow.keras import layers


from modules import constants
from modules.FEN_converter import FENConverter
from modules.autoencoder.base_autoencoder import BaseAutoEncoder

LATENT_DIM = 16
INPUT_SIZE = constants.VECTOR_LENGTH


class Autoencoder_Linear16(BaseAutoEncoder):

    def __init__(self):
        super(Autoencoder_Linear16, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(INPUT_SIZE,)),
                layers.Dense(LATENT_DIM),
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
        self.build((None, self.input_size))
        super(Autoencoder_Linear16, self).load_weights(filepath)

    def vectorize_FEN(self, fen: str):
        return FENConverter.to_vector(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.vector_to_FEN(vector)
