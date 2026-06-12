import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from prepare_data import prepare_data


def get_batch(encoded_text, batch_size, seq_length):
    """Sample a batch of sequences from the encoded text without building all sequences.

    Returns:
        x: torch.LongTensor of shape (batch_size, seq_length)
        y: torch.LongTensor of shape (batch_size*seq_length,)
    """
    total_chars = len(encoded_text)
    max_start = total_chars - seq_length - 1
    if max_start <= 0:
        raise ValueError("Encoded text too short for given seq_length")

    starts = [random.randint(0, max_start) for _ in range(batch_size)]

    x_batch = [encoded_text[s : s + seq_length] for s in starts]
    y_batch = [encoded_text[s + 1 : s + seq_length + 1] for s in starts]

    x = torch.tensor(x_batch, dtype=torch.long)
    y = torch.tensor(y_batch, dtype=torch.long).view(-1)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["lstm", "transformer"]) 
    args = parser.parse_args()

    model_name = args.model

    # Load data
    vocab_size, char_to_int, int_to_char, encoded = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    epochs = 1
    batch_size = 64
    seq_length = 100
    steps_per_epoch = 200
    lr = 0.001

    # Model selection
    if model_name == "lstm":
        from model_lstm import LSTMModel

        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            n_layers=2,
        )
    else:
        from model_transformer import TransformerModel

        model = TransformerModel(
            vocab_size=vocab_size,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            ff_dim=512,
        )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"Training Model: {model_name}\n")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}\n")
        for step in range(1, steps_per_epoch + 1):
            model.train()

            x, y = get_batch(encoded, batch_size, seq_length)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if model_name == "lstm":
                # initialize hidden state per batch
                hidden = model.init_hidden(batch_size)
                # move hidden to device
                hidden = (hidden[0].to(device), hidden[1].to(device))
                output, hidden = model(x, hidden)
            else:
                output = model(x)

            # output: (batch*seq_length, vocab_size)
            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)

            if step % 50 == 0:
                print(f"Step {step} Loss: {loss_value:.4f}\n")

    print("Training Complete\n")

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("saved", exist_ok=True)

    model_path = os.path.join("models", f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)

    loss_path = os.path.join("saved", f"{model_name}_loss.npy")
    np.save(loss_path, np.array(losses))

    print("Model Saved:")
    print(model_path)
    print("\nLoss Saved:")
    print(loss_path)


if __name__ == "__main__":
    main()
