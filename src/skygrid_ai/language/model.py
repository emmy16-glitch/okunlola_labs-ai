
import math

import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_size):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Learn Query, Key and Value projections.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)

        # Stop attention from looking into the future.
        mask = torch.tril(
            torch.ones(
                context_size,
                context_size,
                dtype=torch.bool
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False
        )

    def forward(self, x):
        batch, length, width = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def divide_heads(t):
            return t.reshape(
                batch, length,
                self.n_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = map(divide_heads, (q, k, v))

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~self.causal_mask[:length, :length],
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)

        output = weights @ v

        output = (
            output.transpose(1, 2)
            .contiguous()
            .reshape(batch, length, width)
        )

        return self.projection(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_size):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = CausalSelfAttention(
            d_model, n_heads, context_size
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class SkyGridLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=32,
        d_model=48,
        n_heads=4,
        n_layers=2
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        self.position_embedding = nn.Embedding(
            context_size, d_model
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                d_model, n_heads, context_size
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.output_head = nn.Linear(
            d_model, vocab_size
        )

    def forward(self, token_ids, targets=None):
        _, length = token_ids.shape

        if length > self.context_size:
            raise ValueError(
                "Text exceeds model context size"
            )

        positions = torch.arange(
            length,
            device=token_ids.device
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        logits = self.output_head(
            self.norm(self.blocks(x))
        )

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss
