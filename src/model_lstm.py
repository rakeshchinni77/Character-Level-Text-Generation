import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Character-level LSTM for text generation.

    Architecture: Embedding -> LSTM -> Linear
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        """Initialize LSTMModel.

        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of character embeddings
            hidden_dim (int): Hidden dimension of LSTM
            n_layers (int): Number of LSTM layers
        """
        super(LSTMModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding layer: converts character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: processes sequences with gating mechanism
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers, batch_first=True
        )

        # Linear layer: projects hidden states to vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            hidden (tuple): Tuple of (h0, c0) LSTM hidden states

        Returns:
            output (torch.Tensor): Output logits of shape (batch_size*seq_length, vocab_size)
            hidden (tuple): Updated hidden states (h, c)
        """
        batch_size = x.size(0)

        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embeds = self.embedding(x)

        # LSTM: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Flatten output for linear layer
        # (batch_size, seq_length, hidden_dim) -> (batch_size*seq_length, hidden_dim)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Linear layer: (batch_size*seq_length, hidden_dim) -> (batch_size*seq_length, vocab_size)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden and cell states for LSTM.

        Args:
            batch_size (int): Batch size

        Returns:
            tuple: (h0, c0) where each has shape (n_layers, batch_size, hidden_dim)
        """
        device = next(self.parameters()).device
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


if __name__ == "__main__":
    # Local testing
    model = LSTMModel(vocab_size=65, embedding_dim=128, hidden_dim=256, n_layers=2)

    # Generate random input
    x = torch.randint(0, 65, (32, 100))

    # Initialize hidden state
    hidden = model.init_hidden(32)

    # Forward pass
    output, hidden = model(x, hidden)

    # Print shapes
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Expected Output Shape: (3200, 65)")
    print()
    print("Test Passed!" if output.shape == torch.Size([3200, 65]) else "Test Failed!")
