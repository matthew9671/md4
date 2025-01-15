"""
Jax implementation of the "hollow" version
of the LLAMA2-like Transformer, based on the MD4 implementation
"""

import dataclasses
import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from md4.networks.transformer import RMSNorm, FeedForward

@dataclasses.dataclass(unsafe_hash=True)
class ModelArgs:
  dim: int = 288
  n_layers: int = 6
  n_heads: int = 6
  n_kv_heads: Optional[int] = None
  output_channels: int = 1024
  hidden_dim: Optional[int] = None
  multiple_of: int = 32  # MLP hidden layer size will be multiple of
  norm_eps: float = 1e-5
  dropout_rate: float = 0.0
  weight_tying: bool = False
  w_init_scale: float = 1.0
  depth_scaled_init: bool = False
  # glu, geglu, swiglu
  mlp_type: str = 'swiglu'
  # adaln, adaln_zero
  cond_type: str = 'adaln'
  embed_input: bool = False
  n_embed_classes: int = 1024
  causal: bool = False
  # ----------------------------
  n_layers_per_mixed: int = 2

class MaskedAttention(nn.Module):
  """
  Attention that takes in an additive mask
  Everything else is identical with the original MD4 implementation
  """

  dim: int
  n_heads: int
  n_kv_heads: int | None = None
  dropout_rate: float = 0.0
  causal: bool = False  # Not used
  qkv_bias: bool = False

  def setup(self):
    self._n_kv_heads = (
        self.n_heads if self.n_kv_heads is None else self.n_kv_heads
    )
    assert self.n_heads % self._n_kv_heads == 0
    self.n_rep = self.n_heads // self._n_kv_heads
    self.head_dim = self.dim // self.n_heads
    self.wq = nn.Dense(self.n_heads * self.head_dim, use_bias=self.qkv_bias)
    self.wk = nn.Dense(self._n_kv_heads * self.head_dim, use_bias=self.qkv_bias)
    self.wv = nn.Dense(self._n_kv_heads * self.head_dim, use_bias=self.qkv_bias)
    self.wo = nn.Dense(self.dim, use_bias=False)
    if self.dropout_rate > 0.0:
      self.attn_dropout = nn.Dropout(self.dropout_rate)
      self.resid_dropout = Dropout1d(self.dropout_rate)

  def __call__(self, x, freqs_cos, freqs_sin, attn_mask, train=False):
    bsz, seqlen, _ = x.shape

    # QKV
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)
    xv = xv.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)

    # RoPE relative positional embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

    # grouped multiquery attention: expand out keys and values
    xk = repeat_kv(xk, self.n_rep)
    xv = repeat_kv(xv, self.n_rep)

    # make heads into a batch dimension
    xq = xq.swapaxes(1, 2)  # (bs, n_heads, seqlen, head_dim)
    xk = xk.swapaxes(1, 2)
    xv = xv.swapaxes(1, 2)

    scores = jnp.matmul(xq, xk.swapaxes(2, 3)) / math.sqrt(self.head_dim)

    # Assuming attn_mask has shape (seqlen, seqlen)
    attn_mask = attn_mask[None, None]
    scores = (
          scores + mask[:, :, :seqlen, :seqlen]
      )  # (bs, n_heads, seqlen, seqlen)

    scores = nn.softmax(scores, axis=-1)
    if self.dropout_rate > 0.0:
      scores = self.attn_dropout(scores, deterministic=not train)
    output = jnp.matmul(scores, xv)  # (bs, n_heads, seqlen, head_dim)

    # restore time as batch dimension and concat heads
    output = output.swapaxes(1, 2).reshape(bsz, seqlen, -1)

    # final projection into the residual stream
    output = self.wo(output)
    if self.dropout_rate > 0.0:
      output = self.resid_dropout(output, deterministic=not train)
    return output

class MaskedTransformerBlock(nn.Module):
  """
  Exactly the same as in the MD4 implementation
  Except that it uses masked attention instead of regular attention.
  """

  layer_id: int
  args: ModelArgs

  def setup(self):
    args = self.args
    self.attention = MaskedAttention(
        args.dim,
        args.n_heads,
        n_kv_heads=args.n_kv_heads,
        dropout_rate=args.dropout_rate,
        causal=args.causal,
    )

    if args.depth_scaled_init:
      w_init_scale = 2.0 / args.n_layers
    else:
      w_init_scale = args.w_init_scale

    self.feed_forward = FeedForward(
        dim=args.dim,
        multiple_of=args.multiple_of,
        dropout_rate=args.dropout_rate,
        hidden_dim=args.hidden_dim,
        w_init_scale=w_init_scale,
        mlp_type=args.mlp_type,
    )

  @nn.compact
  def __call__(self, x, freqs_cos, freqs_sin, attn_mask, cond=None, train=False):
    if cond is not None:
      activation = activation_map[self.args.mlp_type]
      if self.args.cond_type == 'adaln':
        ln = nn.Sequential([
            # nn.swish,
            activation,
            nn.Dense(6 * self.args.dim, use_bias=True),
        ])
      elif self.args.cond_type == 'adaln_zero':
        ln = nn.Sequential([
            # nn.swish,
            activation,
            nn.Dense(
                6 * self.args.dim,
                use_bias=True,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            ),
        ])
      else:
        raise NotImplementedError()
      (shift_att, scale_att, gate_att, shift_mlp, scale_mlp, gate_mlp) = (
          jnp.split(ln(cond)[:, None, :], 6, axis=-1)
      )
      attention_norm = nn.LayerNorm(
          epsilon=self.args.norm_eps, use_bias=False, use_scale=False
      )
      ffn_norm = nn.LayerNorm(
          epsilon=self.args.norm_eps, use_bias=False, use_scale=False
      )
      h = x + gate_att * self.attention(
          attention_norm(x) * (scale_att + 1.0) + shift_att,
          freqs_cos,
          freqs_sin,
          attn_mask,
          train=train,
      )
      out = h + gate_mlp * self.feed_forward(
          ffn_norm(h) * (scale_mlp + 1.0) + shift_mlp, train=train
      )
    else:
      attention_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
      ffn_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
      h = x + self.attention(
          attention_norm(x), freqs_cos, freqs_sin, train=train
      )
      out = h + self.feed_forward(ffn_norm(h), train=train)

    return out


class HollowTransformer(nn.Module):

  args: ModelArgs

  @nn.compact
  def __call__(self, x, cond=None, train=False, output_channels=None):
    args = self.args
    if output_channels is None:
      output_channels = args.output_channels

    # Pad x on both sides by 0
    x = jnp.insert(x, 0, 0, axis=1)
    x = jnp.insert(x, x.shape[1], 0, axis=1)

    if args.embed_input:
      h = nn.Embed(args.n_embed_classes, args.dim)(x)
      if args.dropout_rate > 0.0:
        h = nn.Dropout(args.dropout_rate, deterministic=not train)(h)
    else:
      h = nn.Dense(args.dim)(x)

    seqlen = x.shape[1]
    freqs_cos, freqs_sin = precompute_freqs_cis(
        args.dim // args.n_heads, seqlen
    )

    freqs_cos = freqs_cos[:seqlen]
    freqs_sin = freqs_sin[:seqlen]

    # Offset the two streams and initialize the mixed stream to None
    freqs_cos_f = freqs_cos[:-2]
    freqs_sin_f = freqs_sin[:-2]
    freqs_cos_b = freqs_cos[2:]
    freqs_sin_b = freqs_sin[2:]
    freqs_cos_m = freqs_cos[1:-1]
    freqs_sin_m = freqs_sin[1:-1]
    hf = h[:,:-2]
    hb = h[:,2:]
    # Use sum instead of concatenation so we can keep dimension constant
    hm = hf + hb #jnp.concatenate([hf, hb], axis=2)
    mask = jnp.full((L, L), -jnp.inf)
    forward_mask = jnp.tril(mask)
    backward_mask = jnp.triu(mask)
    mixing_mask = jnp.concatenate([forward_mask, backward_mask], axis=-1)   

    layer_id = 0

    for layer in range(args.n_layers):
      # Forward stream
      hf = MaskedTransformerBlock(layer_id, args)(
          hf, freqs_cos_f, freqs_sin_f, forward_mask, cond=cond, train=train
      )
      layer_id += 1
      # Backward stream
      hb = MaskedTransformerBlock(layer_id, args)(
          hb, freqs_cos_b, freqs_sin_b, backward_mask, cond=cond, train=train
      )
      layer_id += 1
      # Mixing stream
      if (layer + 1) % args.n_layers_per_mixed == 0:
        hfb = jnp.concatenate([hf, hb], axis=1)
        hm = MaskedTransformerBlock(layer_id, args)(
          hm, freqs_cos_m, freqs_sin_m, mixing_mask, cond=cond, train=train
        )
        layer_id += 1

    # Use the last mixing stream output
    h = hm

    if cond is not None:
      output_norm = nn.LayerNorm(
          epsilon=args.norm_eps, use_bias=False, use_scale=False
      )
      activation = activation_map[args.mlp_type]
      if args.cond_type == 'adaln':
        ln = nn.Sequential([
            # nn.swish,
            activation,
            nn.Dense(2 * args.dim, use_bias=True),
        ])
      elif args.cond_type == 'adaln_zero':
        ln = nn.Sequential([
            # nn.swish,
            activation,
            nn.Dense(
                2 * args.dim,
                use_bias=True,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            ),
        ])
      else:
        raise NotImplementedError()
      shift_out, scale_out = jnp.split(ln(cond)[:, None, :], 2, axis=-1)
      logits = nn.Dense(
          output_channels, use_bias=False, kernel_init=nn.initializers.zeros
      )(output_norm(h) * (scale_out + 1) + shift_out)
    else:
      h = RMSNorm(args.dim, args.norm_eps)(h)
      logits = nn.Dense(
          features=output_channels,
          use_bias=False,
          kernel_init=nn.initializers.zeros,
      )(h)

    return logits
