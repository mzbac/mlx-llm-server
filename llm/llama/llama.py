from typing import Optional, Tuple
import math
import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs


class LinearScalingRoPE(nn.RoPE):
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        rope_scaling_factor: float = 1.0,
        base: float = 10000,
        dtype=mx.float32,
    ):
        super().__init__(dims)
        self.base = base
        self.rope_scaling_factor = rope_scaling_factor

        # according to the paper, the head dimension should be even
        assert dims % 2 == 0, "dims must be divisble by 2"

        D = dims // 2
        self.freqs = mx.exp(-mx.arange(0.0, D, dtype=dtype) * (math.log(self.base) / D))
        self._set_cos_sin_cache(max_position_embeddings, dtype=dtype)

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        if N > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=N, dtype=x.dtype
            )  # in case the default max_position_embeddings is incorret, make sure the backward compatibility

        rx = self._compute_rope(self.costheta[offset:N], self.sintheta[offset:N], x)

        return mx.reshape(rx, shape)

    def _set_cos_sin_cache(self, seq_len: int, dtype=mx.float32):
        self.max_seq_len_cached = seq_len
        positions = mx.arange(0, self.max_seq_len_cached, dtype=dtype)
        positions = positions / self.rope_scaling_factor
        theta = mx.reshape(positions, (-1, 1)) * mx.reshape(self.freqs, (1, -1))
        self.costheta = mx.cos(theta)
        self.sintheta = mx.sin(theta)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.act_fn = nn.silu

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_attention_heads: int = args.num_attention_heads
        self.num_key_value_heads: int = args.num_key_value_heads
        self.repeats = self.num_attention_heads // self.num_key_value_heads

        self.head_dim = args.hidden_size // args.num_attention_heads

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        if args.rope_scaling is None:
            self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)
        else:
            scaling_factor = args.rope_scaling.get("factor", 1.0)
            self.rope = LinearScalingRoPE(
                self.head_dim,
                max_position_embeddings=args.max_position_embeddings,
                rope_scaling_factor=scaling_factor,
                base=args.rope_theta,
                dtype=args.mlx_dtype,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        k = k.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.num_attention_heads, L, -1])

        k, v = map(repeat, (k, v))

        if cache is not None:
            k_cache, v_cache = cache
            q = self.rope(q, offset=k_cache.shape[2])
            k = self.rope(k, offset=k_cache.shape[2])
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        else:
            q = self.rope(q)
            k = self.rope(k)

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)

        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        v_hat = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)

        return self.o_proj(v_hat), (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = FeedForward(args=args)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        x = self.embed_tokens(x)

        mask = None
        T = x.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(x.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        x = self.norm(x)
        return self.lm_head(x), cache
