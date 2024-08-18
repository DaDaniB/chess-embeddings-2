import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback


class BatchHistory(Callback):
    def on_train_begin(self, logs=None):
        self.batch_loss = []
        self.epoch_val_loss = []

    def on_batch_end(self, batch, logs=None):
        self.batch_loss.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_val_loss.append(logs.get("val_loss"))

    def plot_metrics(self, name):
        plt.figure(figsize=(24, 10))
        plt.subplot(1, 2, 1)
        plt.plot(self.batch_loss, label="Training Loss")
        plt.title("Model Loss (avg. during epoch)")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_val_loss, label="Training Loss")
        plt.title("Model Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.legend(loc="upper right")

        plt.savefig(f"./{name}_metrics.png")
