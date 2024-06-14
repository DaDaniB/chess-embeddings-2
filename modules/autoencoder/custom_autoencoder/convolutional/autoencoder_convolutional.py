import tensorflow as tf
from tensorflow.keras import layers

from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.FEN_converter import FENConverter
from modules import constants


class Autoencoder_Convolutional(BaseAutoEncoder):
    def __init__(self):
        super(Autoencoder_Convolutional, self).__init__()

        first_filter = 128
        second_filter = 128 / 2

        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(8, 8, 12)),
                layers.Conv2D(
                    first_filter,
                    (3, 3),
                    activation="sigmoid",
                    padding="same",
                    strides=2,
                ),
                layers.Conv2D(
                    second_filter,
                    (3, 3),
                    activation="sigmoid",
                    padding="same",
                    strides=2,
                ),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2DTranspose(
                    second_filter,
                    kernel_size=3,
                    strides=2,
                    activation="sigmoid",
                    padding="same",
                ),
                layers.Conv2DTranspose(
                    first_filter,
                    kernel_size=3,
                    strides=2,
                    activation="sigmoid",
                    padding="same",
                ),
                layers.Conv2D(
                    12, kernel_size=(3, 3), activation="sigmoid", padding="same"
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
        super(Autoencoder_Convolutional, self).load_weights(filepath)

    def vectorize_FEN(self, fen: str):
        return FENConverter.to_bitboards(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.bitboards_to_FEN(vector)
