import matplotlib.pyplot as plt


class Autoencoder_Metrics:
    def __init__(self):
        self.accumulated_loss = []
        self.cumulative_inputs = []
        self.epoch_data = []

    def add_history(self, history, num_inputs):
        self.accumulated_loss.extend(history.history["loss"])
        prev_cumulative_inputs = 0

        if self.cumulative_inputs is not None and len(self.cumulative_inputs) > 0:
            prev_cumulative_inputs = self.cumulative_inputs[-1]

        self.cumulative_inputs.append(prev_cumulative_inputs + num_inputs)

    def add_epoch_data(self):
        self.epoch_data.append((self.cumulative_inputs[-1], self.accumulated_loss[-1]))

    def save(self, savename: str):
        plt.figure(figsize=(10, 5))
        plt.plot(self.cumulative_inputs, self.accumulated_loss, label="Training Loss")

        max_loss = max(self.accumulated_loss)
        min_loss = min(self.accumulated_loss)
        middle_loss = ((max_loss - min_loss) / 2) + min_loss

        for epoch in self.epoch_data:
            print(f"epoch {epoch}")
            plt.axvline(x=epoch[0], color="r", linestyle="--")
            plt.text(
                x=epoch[0],
                y=middle_loss,
                s=f"{epoch[1]:.5f}",
                color="r",
                ha="right",
                va="center",
            )

        plt.xlabel("Number of Inputs Processed")
        plt.ylabel("Loss")
        plt.plot([], [], "r--", label="Epochs")
        plt.legend()
        plt.title("Training Loss Over Number of Inputs Processed")
        plt.savefig("./" + savename + "_loss.png")
