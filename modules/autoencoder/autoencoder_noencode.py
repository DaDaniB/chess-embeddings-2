import tensorflow as tf
from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.FEN_converter import FENConverter
from modules.vectorization.FEN import FEN


class Autoencoder_NoEncode(BaseAutoEncoder):

    def __init__(self):
        super(Autoencoder_NoEncode, self).__init__()
        self.encoder = tf.keras.Sequential([])

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

    def load_weights(self, filepath):
        self.build((None, self.input_size))
        super(Autoencoder_NoEncode, self).load_weights(filepath)

    def vectorize_FEN(self, FEN: FEN):
        return FENConverter.to_vector(FEN)

    def vector_to_FEN(self, vector):
        return FENConverter.vector_to_FEN(vector)
