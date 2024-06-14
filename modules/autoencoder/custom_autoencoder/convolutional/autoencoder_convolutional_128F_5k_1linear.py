import tensorflow as tf
from tensorflow.keras import layers

from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.FEN_converter import FENConverter
from modules import constants

NUM_FILTERS = 128
KERNEL_SIZE = 5
LATENT_DIM = 128
ACTIVATION = "relu"
DTYPE = "float32"


class Autoencoder_Convolutional_128F_5K_1linear(BaseAutoEncoder):

    def __init__(self):
        super(Autoencoder_Convolutional_128F_5K_1linear, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                layers.Input(
                    shape=(
                        constants.CONVOLUTIONAL_FIRST_DIM,
                        constants.CONVOLUTIONAL_SECOND_DIM,
                        constants.CONVOLUTIONAL_THIRD_DIM,
                    )
                ),
                layers.Conv2D(
                    NUM_FILTERS,
                    kernel_size=KERNEL_SIZE,
                    padding="same",
                    activation=ACTIVATION,
                    dtype=DTYPE,
                ),
                layers.Conv2D(
                    NUM_FILTERS,
                    kernel_size=KERNEL_SIZE,
                    padding="same",
                    activation=ACTIVATION,
                    dtype=DTYPE,
                ),
                layers.Dense(LATENT_DIM, dtype=DTYPE),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(LATENT_DIM),
                layers.Conv2DTranspose(
                    NUM_FILTERS,
                    kernel_size=KERNEL_SIZE,
                    padding="same",
                    activation=ACTIVATION,
                    dtype=DTYPE,
                ),
                layers.Conv2DTranspose(
                    NUM_FILTERS,
                    kernel_size=KERNEL_SIZE,
                    padding="same",
                    activation=ACTIVATION,
                    dtype=DTYPE,
                ),
                layers.Conv2D(
                    constants.CONVOLUTIONAL_THIRD_DIM,
                    kernel_size=KERNEL_SIZE,
                    padding="same",
                    activation=ACTIVATION,
                    dtype=DTYPE,
                ),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def load_weights(self, filepath):
        self.build(
            (
                None,
                constants.CONVOLUTIONAL_FIRST_DIM,
                constants.CONVOLUTIONAL_SECOND_DIM,
                constants.CONVOLUTIONAL_THIRD_DIM,
            )
        )
        super(Autoencoder_Convolutional_128F_5K_1linear, self).load_weights(filepath)

    def vectorize_FEN(self, fen: str):
        return FENConverter.to_bitboards(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.bitboards_to_FEN(vector)
