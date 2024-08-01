import tensorflow as tf
from tensorflow.keras import layers

from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.FEN_converter import FENConverter
from modules import constants


class Autoencoder_Convolutional_Simple(BaseAutoEncoder):
    def __init__(self):
        super(Autoencoder_Convolutional_Simple, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(8, 8, 12)),
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                ),
                layers.Conv2D(
                    64, (3, 3), activation="relu", padding="same", strides=(2, 2)
                ),
                layers.Conv2D(
                    128, (3, 3), activation="relu", padding="same", strides=(2, 2)
                ),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(2 * 2 * 128, activation="relu"),
                layers.Reshape((2, 2, 128)),
                layers.Conv2DTranspose(
                    128,
                    (3, 3),
                    activation="relu",
                    strides=2,
                    padding="same",
                ),
                layers.Conv2DTranspose(
                    64,
                    (3, 3),
                    activation="relu",
                    strides=2,
                    padding="same",
                ),
                layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"),
                layers.Conv2DTranspose(
                    12, (3, 3), activation="sigmoid", padding="same"
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
        super(Autoencoder_Convolutional_Simple, self).load_weights(filepath)

    def vectorize_FEN(self, fen: str):
        return FENConverter.to_bitboards(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.bitboards_to_FEN(vector)
