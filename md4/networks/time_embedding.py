from jax import numpy as jnp
from flax import linen as nn

class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256

    def setup(self):
        self.mlp = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.silu,
            nn.Dense(self.hidden_size),
        ])

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
        args = t[:, None] * freqs[None]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)