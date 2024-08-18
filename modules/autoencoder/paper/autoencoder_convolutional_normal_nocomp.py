import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from modules.autoencoder.weighted_mse import WeightedMSE
from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.FEN_converter import FENConverter
from modules import constants


class Autoencoder_Convolutional_normal_nocomp(BaseAutoEncoder):
    def __init__(self):
        super(Autoencoder_Convolutional_normal_nocomp, self).__init__()
        initializer = tf.keras.initializers.HeNormal()

        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(8, 8, 12)),
                layers.Conv2D(
                    32, (3, 3), padding="same", kernel_initializer=initializer
                ),
                # BatchNormalization(),
                Activation("relu"),
                Dropout(0.25),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(
                    64, (3, 3), padding="same", kernel_initializer=initializer
                ),
                # BatchNormalization(),
                Activation("relu"),
                Dropout(0.25),
                layers.MaxPooling2D((2, 2), padding="same"),
                # BatchNormalization(),
                Activation("relu"),
                Dropout(0.25),
                layers.Flatten(),
                layers.Dense(256, activation="relu", kernel_initializer=initializer),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Reshape((2, 2, 64)),
                layers.Conv2DTranspose(
                    64,
                    (3, 3),
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                ),
                # BatchNormalization(),
                Activation("relu"),
                Dropout(0.25),
                layers.Conv2DTranspose(
                    32,
                    (3, 3),
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                ),
                # BatchNormalization(),
                Activation("relu"),
                Dropout(0.25),
                layers.Conv2DTranspose(
                    12,
                    (3, 3),
                    activation="sigmoid",
                    padding="same",
                    kernel_initializer=initializer,
                ),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def initialize(self):
        optimizer = Adam(learning_rate=1e-4)
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
        super(Autoencoder_Convolutional_normal_nocomp, self).load_weights(filepath)

    def load_default_weights(self):
        self.load_weights(
            "./models/Autoencoder_Convolutional_normal_nocompnPositions3000000.h5"
        )

    def vectorize_FEN(self, fen: str):
        return FENConverter.to_bitboards(fen)

    def vector_to_FEN(self, vector):
        return FENConverter.bitboards_to_FEN(vector)
