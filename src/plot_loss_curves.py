import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Check if loss files exist
    lstm_loss_path = os.path.join("saved", "lstm_loss.npy")
    transformer_loss_path = os.path.join("saved", "transformer_loss.npy")

    if not os.path.exists(lstm_loss_path):
        print(f"Error: {lstm_loss_path} not found")
        return

    if not os.path.exists(transformer_loss_path):
        print(f"Error: {transformer_loss_path} not found")
        return

    # Load loss files
    lstm_losses = np.load(lstm_loss_path)
    transformer_losses = np.load(transformer_loss_path)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot losses
    plt.plot(lstm_losses, label="LSTM", linewidth=2)
    plt.plot(transformer_losses, label="Transformer", linewidth=2)

    # Set labels and title
    plt.title("Training Loss Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Tight layout
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)

    # Save figure
    output_path = os.path.join("results", "loss_curves.png")
    plt.savefig(output_path)

    # Print output
    print("Loss Curve Saved:")
    print(output_path)


if __name__ == "__main__":
    main()
