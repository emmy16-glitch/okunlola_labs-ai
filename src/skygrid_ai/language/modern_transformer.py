
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        rms = torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True)
            + self.eps
        )
        return x * rms * self.weight


def apply_rope(q, k):
    """Apply rotary positions to queries and keys."""

    head_dim = q.size(-1)

    if head_dim % 2:
        raise ValueError(
            "RoPE needs an even attention-head dimension"
        )

    positions = torch.arange(
        q.size(-2),
        device=q.device,
        dtype=torch.float32
    )

    frequency = 1.0 / (
        10000.0 ** (
            torch.arange(
                0, head_dim, 2,
                device=q.device,
                dtype=torch.float32
            ) / head_dim
        )
    )

    angles = torch.outer(positions, frequency)

    cosine = angles.cos()[None, None, :, :].to(
        dtype=q.dtype
    )

    sine = angles.sin()[None, None, :, :].to(
        dtype=q.dtype
    )

    def rotate(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]

        return torch.stack(
            (
                even * cosine - odd * sine,
                even * sine + odd * cosine
            ),
            dim=-1
        ).flatten(-2)

    return rotate(q), rotate(k)


class CausalAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, d_model, n_heads):
        super().__init__()

        if (
            d_model % n_heads
            or (d_model // n_heads) % 2
        ):
            raise ValueError(
                "Attention-head dimension must be even"
            )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(
            d_model, 3 * d_model, bias=False
        )

        self.projection = nn.Linear(
            d_model, d_model, bias=False
        )

    def forward(self, x):
        batch, length, width = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(t):
            return t.reshape(
                batch,
                length,
                self.n_heads,
                self.head_dim
            ).transpose(1, 2)

        q = heads(q)
        k = heads(k)
        v = heads(v)

        q, k = apply_rope(q, k)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=0.0
        )

        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(batch, length, width)
        )

        return self.projection(attended)


class SwiGLU(nn.Module):
    """Gated feed-forward network."""

    def __init__(self, d_model, hidden_width):
        super().__init__()

        self.gate = nn.Linear(
            d_model, hidden_width, bias=False
        )

        self.up = nn.Linear(
            d_model, hidden_width, bias=False
        )

        self.down = nn.Linear(
            hidden_width, d_model, bias=False
        )

    def forward(self, x):
        gated = F.silu(self.gate(x)) * self.up(x)
        return self.down(gated)


class TransformerBlockV08(nn.Module):
    def __init__(
        self, d_model, n_heads, hidden_width
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = CausalAttention(
            d_model, n_heads
        )

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(
            d_model, hidden_width
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SkyGridTransformerV08(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=128,
        d_model=96,
        n_heads=4,
        n_layers=3,
        hidden_width=256
    ):
        super().__init__()

        if (
            vocab_size < 258
            or context_size < 1
            or n_layers < 1
        ):
            raise ValueError(
                "Invalid vocabulary, context, or layer count"
            )

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlockV08(
                d_model,
                n_heads,
                hidden_width
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        self.output_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

        # Share input and output token weights.
        self.output_head.weight = (
            self.token_embedding.weight
        )

        # Small initialization prevents very large
        # initial logits when embeddings are tied.
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02
        )

    def forward(self, token_ids, targets=None):
        if (
            token_ids.ndim != 2
            or not 1 <= token_ids.size(1) <= self.context_size
        ):
            raise ValueError(
                "Token IDs need shape "
                "[batch, 1..context_size]"
            )

        x = self.token_embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        logits = self.output_head(
            self.final_norm(x)
        )

        loss = None

        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError(
                    "Targets must match input shape"
                )

            loss = F.cross_entropy(
                logits.reshape(
                    -1, logits.size(-1)
                ),
                targets.reshape(-1)
            )

        return logits, loss
