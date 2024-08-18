import tensorflow as tf


class WeightedMSE(tf.keras.losses.Loss):
    def __init__(self, weight_for_1=10.0, weight_for_0=1.0, name="weighted_mse"):
        super().__init__(name=name)
        self.weight_for_1 = weight_for_1
        self.weight_for_0 = weight_for_0

    def call(self, y_true, y_pred):
        # Calculate the squared differences
        squared_diff = tf.square(y_true - y_pred)

        # Apply different weights depending on whether y_true is 1 or 0
        weights = tf.where(y_true == 1, self.weight_for_1, self.weight_for_0)

        # Apply the weights to the squared differences
        weighted_squared_diff = weights * squared_diff

        # Calculate the mean of the weighted squared differences
        loss = tf.reduce_mean(weighted_squared_diff)

        return loss
