import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            # odd embedding: last column stays zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.dtype)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Shape to (batch, num_heads, seq_len, head_dim)
        def shape(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape(q)
        k = shape(k)
        v = shape(v)

        # Scaled dot-product attention
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        # context: (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attn, v)

        # Concatenate heads -> (batch, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        out = self.out_proj(context)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + Add & Norm
        attn_out = self.self_attn(x)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward + Add & Norm
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)

        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, ff_dim: int, max_len: int = 5000):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)

        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()

        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        # x: (batch, seq_len, embed_dim)
        x = x.contiguous().view(batch_size * seq_len, self.embed_dim)
        out = self.fc_out(x)
        return out


if __name__ == "__main__":
    # Local test
    model = TransformerModel(vocab_size=65, embed_dim=128, num_heads=4, num_layers=2, ff_dim=512)
    x = torch.randint(0, 65, (32, 100))
    out = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Expected Output Shape: (3200, 65)")
    print("Test Passed!" if out.shape == torch.Size([3200, 65]) else "Test Failed!")
