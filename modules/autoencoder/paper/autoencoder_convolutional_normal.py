import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from modules.autoencoder.weighted_mse import WeightedMSE
from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.FEN_converter import FENConverter
from modules import constants


class Autoencoder_Convolutional_normal(BaseAutoEncoder):
    def __init__(self):
        super(Autoencoder_Convolutional_normal, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(8, 8, 12)),
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                ),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.Reshape((2, 2, 64)),
                layers.Conv2DTranspose(
                    64,
                    (3, 3),
                    activation="relu",
                    strides=2,
                    padding="same",
                ),
                layers.Conv2DTranspose(
                    32, (3, 3), activation="relu", strides=2, padding="same"
                ),
                layers.Conv2DTranspose(
                    12, (3, 3), activation="sigmoid", padding="same"
                ),
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
        self.build(
            (
                None,
                constants.CONVOLUTIONAL_FIRST_DIM,
                constants.CONVOLUTIONAL_SECOND_DIM,
                constants.CONVOLUTIONAL_THIRD_DIM,
            )
        )
        super(Autoencoder_Convolutional_normal, self).load_weights(filepath)

    def load_default_weights(self):
        self.load_weights(
            "./models/Autoencoder_Convolutional_normalnPositions3000000.h5"
        )

    def vectorize_FEN(self, fen: str):
        return FENConverter.to_bitboards(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.bitboards_to_FEN(vector)
