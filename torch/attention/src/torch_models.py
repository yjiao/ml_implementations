"""
Implementation of attention-related modules in pytorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import math
import numpy as np

from collections import OrderedDict

from typing import Type


class PositionalEncodingInterface(nn.Module):
    """Base class for positional embeddings to ensure variants use the same interface.

    Do not use directly as forward function is not implemented.
    """

    def __init__(self, d_model: int, max_context: int):
        super().__init__()


@dataclass
class ModelConfig:
    dim: int  # dimension of embeddings and encoder/ decoder outputs
    N: int  # number of encoders and decoders
    h: int  # number of heads
    V: int  # vocab size
    max_context: int
    drop_rate: float = 0.1
    verbose: bool = False
    position_encoding_class: Type[nn.Module] = PositionalEncodingInterface


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embeddings = nn.Linear(vocab_size, embedding_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, X: torch.tensor) -> torch.tensor:
        select = F.one_hot(X, num_classes=self.vocab_size).type(torch.float32)
        return self.embeddings(select)


class LearnablePositionalEncoding(PositionalEncodingInterface):
    def __init__(self, d_model: int, max_context: int):
        super().__init__(d_model, max_context)

        self.embeddings = nn.Parameter(torch.zeros(1, max_context, d_model))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.embeddings[:, 0 : x.size(1)]


class HardcodedPositionalEncoding(PositionalEncodingInterface):
    # modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, max_context: int):
        super().__init__(d_model, max_context)

        position = torch.arange(max_context).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.div_term = div_term
        self.position = position
        pe = torch.zeros(1, max_context, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, 0 : x.size(1)]


class Res(nn.Module):
    """Residual block using zero-padding."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        # apply zero padding for no added parameters
        f_x = self.module(*args)
        X = args[0]
        pad_needed = (0, f_x.shape[-1] - X.shape[-1])
        padded = torch.nn.functional.pad(X, pad_needed)
        return padded + f_x


class SelfAttentionMulti(nn.Module):
    """
    Performs multi-headed self attention with optional masking.

    Does not implement the feed forward layers after attention.

    Input dimension == output dimension.
    """

    def __init__(
        self,
        n_heads: int,
        dim_io: int,
        max_context_size: int,
        causal: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose

        # assume k, q, v have same inner dimension. This is usually the case in practice
        assert dim_io % n_heads == 0, (
            "input dimension must be divisible by the number of heads. "
            + f"Got {dim_io} input dimension and {n_heads} heads."
        )

        self.n_heads = n_heads
        self.dim_io = dim_io
        self.dim_per_head = dim_io // n_heads

        self.Wk = nn.Linear(dim_io, dim_io, bias=False)
        self.Wq = nn.Linear(dim_io, dim_io, bias=False)
        self.Wv = nn.Linear(dim_io, dim_io, bias=False)

        # if causal attention is needed, add a tril mask
        self.causal = causal
        if causal:
            mask = torch.ones(max_context_size, max_context_size)
            mask = torch.tril(mask)
            # This is typically used to register a buffer that should not to be
            # considered a model parameter
            self.register_buffer("mask", mask)

        self.proj = nn.Linear(dim_io, dim_io)

        if self.verbose:
            self.init_print()

    def attention(
        self, k: torch.tensor, q: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        kq = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(k.shape[-1])
        if self.causal:
            seqlen = kq.shape[-2]
            kq.masked_fill_(self.mask[:seqlen, :seqlen] == 0, -1e9)
        kq = nn.functional.softmax(kq, dim=-1)
        if self.verbose:
            print("    >>>>> attention method")
            print(f"    kq after masking: (causal = {self.causal})")
            print("       k", k.shape)
            print("       q", q.shape)
            print("       kq", kq.shape)
        return torch.matmul(kq, v)

    def multihead_attention(
        self, k: torch.tensor, q: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        """Given flat k, q, v, stack and perform multi-headed attention."""
        kstack = self.stack_heads(k)
        qstack = self.stack_heads(q)
        vstack = self.stack_heads(v)

        attn = self.attention(kstack, qstack, vstack)
        return self.proj(self.unstack_heads(attn))

    def forward(self, X: torch.tensor) -> torch.tensor:
        # X: (batch, seqlen, dim_io)
        # W: (dim_io, dim_io)
        # dim_io = dim_per_head * n_heads
        if self.verbose:
            print(">>>>> SelfAttentionMulti forward method")
            print("    X shape:", tuple(X.shape))
            print("    W_k shape:", tuple(self.Wk.weight.shape))

        k = self.Wk(X)  # output shape: (batch, seqlen, dim_io)
        q = self.Wq(X)
        v = self.Wv(X)

        return self.multihead_attention(k, q, v)

    def stack_heads(self, flat: torch.tensor) -> torch.tensor:
        """Convert attention head values from vcat to stacked.

        (batch, seqlen, n_heads * dim_per_head) -> (batch, n_heads, seqlen, dim_per_head)
        """
        batch, seqlen, _ = flat.shape
        target_shape = (batch, seqlen, self.n_heads, self.dim_per_head)
        return flat.view(target_shape).transpose(1, 2)

    def unstack_heads(self, stacked: torch.tensor) -> torch.tensor:
        """Convert attention head values from stacked to vertically concatenated.

        (batch, n_heads, seqlen, dim_per_head) -> (batch, seqlen, n_heads * dim_per_head)
        """
        batch, _, seqlen, _ = stacked.shape
        stacked = stacked.transpose(1, 2).contiguous()
        return stacked.view((batch, seqlen, self.dim_io))

    def init_print(self):
        print(
            f"""
-------------------------------------------------------------------
Creating Multi-headed self-attention module with:
    Casual attention: {self.causal}
    {self.n_heads} heads x dim [{self.dim_per_head}]
    [{self.dim_io}] input and output size
-------------------------------------------------------------------
"""
        )


class CrossAttentionMulti(SelfAttentionMulti):
    """Implements encoder-decoder attention."""

    def forward(self, dec_input: torch.tensor, enc_input: torch.tensor) -> torch.tensor:
        # dec_input: (batch, seqlen, emb)
        # W: (emb, dim_kq * n_heads)
        k = self.Wk(enc_input)
        v = self.Wv(enc_input)
        q = self.Wq(dec_input)
        attn = self.multihead_attention(k, q, v)

        if self.verbose:
            print(">>>>> CrossAttentionMulti forward method")
            print("    Decoder input shape:", tuple(dec_input.shape))
            print("    Encoder input shape:", tuple(enc_input.shape))
            print("    W_k shape:", tuple(self.Wk.weight.shape))
            print(f"    k shape:    {tuple(k.shape)})")
            print(f"    q shape:    {tuple(q.shape)})")
            print(f"    v shape:    {tuple(v.shape)})")
            print(f"    attn shape: {tuple(attn.shape)})")

        return attn

    def init_print(self):
        print(
            f"""
-------------------------------------------------------------------
Creating Multi-headed encoder-decoder attention module with:
    Casual attention: {self.causal}
    {self.n_heads} heads x dim [{self.dim_per_head}]
    [{self.dim_io}] input and output size
-------------------------------------------------------------------
"""
        )


class Encoder(nn.Module):
    """Simple encoder implementation.

    From paper: "We apply dropout [33] to the output of each sub-layer, before it is added to the
        sub-layer input and normalized." Take this to mean dropout occurs inside the res block.
    """

    def __init__(
        self,
        n_heads: int,
        dim_io: int,
        max_context_size: int,
        droprate: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__()

        # dimension in must = dimension out
        assert dim_io % n_heads == 0, (
            "input dimension must be divisible by the number of heads. "
            + f"Got {dim_io} input dimension and {n_heads} heads."
        )

        self.attn_norm = torch.nn.LayerNorm(dim_io)
        self.attn = SelfAttentionMulti(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            causal=False,
            verbose=verbose,
        )
        self.drop = nn.Dropout(droprate)

        # inner ff has 2048, output is 512
        dim_ff = 2048
        self.ff_norm = torch.nn.LayerNorm(dim_io)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_io, dim_ff),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(dim_ff, dim_io),
            nn.Dropout(droprate),
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        attn = self.attn_norm(X)
        attn = self.attn(attn)
        attn = self.drop(attn)
        attn = X + attn  # manual residual

        # ff
        out = self.ff_norm(attn)
        out = self.feed_forward(out)
        out = attn + out
        return out


class Decoder(nn.Module):
    """Decoder module with encoder-decoder attention."""

    def __init__(
        self,
        n_heads: int,
        dim_io: int,
        max_context_size: int,
        droprate: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose

        # dimension in must = dimension out
        assert dim_io % n_heads == 0, (
            "input dimension must be divisible by the number of heads. "
            + f"Got {dim_io} input dimension and {n_heads} heads."
        )

        self.ln1 = nn.LayerNorm(dim_io)
        self.drop = nn.Dropout(droprate)
        self.causal_attn = SelfAttentionMulti(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            causal=True,
            verbose=verbose,
        )

        self.ln2 = torch.nn.LayerNorm(dim_io)
        self.cross_attn = CrossAttentionMulti(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            causal=False,
            verbose=verbose,
        )

        # from paper: inner ff has 2048, output is 512
        dim_inner_ff = 2048
        self.ln3 = torch.nn.LayerNorm(dim_io)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_io, dim_inner_ff),
            nn.ReLU(),
            self.drop,
            nn.Linear(dim_inner_ff, dim_io),
            self.drop,
        )

    def forward(self, dec_input: torch.tensor, enc_input: torch.tensor) -> torch.tensor:
        dec = self.ln1(dec_input)
        dec = self.causal_attn(dec)
        dec = self.drop(dec)
        dec = dec_input + dec  # residual

        enc_input = self.ln2(enc_input)
        cross = self.cross_attn(dec, enc_input)
        cross = self.drop(cross)
        # note: from paper figure 1, identity branch = dec
        cross = dec + cross  # residual

        out = self.ln3(cross)
        out = self.feed_forward(out)
        out = cross + out  # residual
        return out


class AutoregressiveDecoder(nn.Module):
    """Decoder module with causal self-attention only, no encoder-decoder attention."""

    def __init__(
        self,
        n_heads: int,
        dim_io: int,
        max_context_size: int,
        droprate: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose

        # dimension in must = dimension out
        assert dim_io % n_heads == 0, (
            "input dimension must be divisible by the number of heads. "
            + f"Got {dim_io} input dimension and {n_heads} heads."
        )

        self.ln1 = nn.LayerNorm(dim_io)
        self.drop = nn.Dropout(droprate)
        self.causal_attn = SelfAttentionMulti(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            causal=True,
            verbose=verbose,
        )

        # from paper: inner ff has 2048, output is 512
        dim_inner_ff = 2048
        self.ln2 = torch.nn.LayerNorm(dim_io)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_io, dim_inner_ff),
            nn.ReLU(),
            self.drop,
            nn.Linear(dim_inner_ff, dim_io),
            self.drop,
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        dec = self.ln1(X)
        dec = self.causal_attn(dec)
        dec = self.drop(dec)
        dec = X + dec  # residual

        out = self.ln2(dec)
        out = self.feed_forward(out)
        out = dec + out  # residual
        return out


class AttnIsAllYouNeed(nn.Module):
    """Simple implementation of the architecture described in the Attention is All You Need paper."""

    def __init__(
        self,
        emb_dimension: int,
        attn_dimension: int,
        vocab_size: int,
        max_context_size: int,
        num_encoders: int,
        num_decoders: int,
        n_heads: int,
        droprate: float = 0.1,
        position_encoding_class: Type[nn.Module] = HardcodedPositionalEncoding,
        verbose: bool = False,
    ):
        assert (
            attn_dimension % n_heads == 0
        ), f"attn_dimension must be divisible by n_head. Got {attn_dimension} and {n_heads}."
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_dimension)
        self.position_encoder = position_encoding_class(
            emb_dimension,
            max_context=max_context_size,
        )
        self.emb_drop = nn.Dropout(droprate)

        self.encoders = nn.Sequential(
            *[
                Encoder(
                    n_heads,
                    attn_dimension,
                    max_context_size,
                    droprate=droprate,
                    verbose=verbose,
                )
                for _ in range(num_encoders)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    n_heads,
                    attn_dimension,
                    max_context_size,
                    droprate=droprate,
                    verbose=verbose,
                )
                for _ in range(num_decoders)
            ]
        )

        self.linear = nn.Linear(attn_dimension, vocab_size)

        # caching
        self.cache_encoder_output = False
        self.prev_enc = None

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(
            emb_dimension=config.dim,
            attn_dimension=config.dim,
            vocab_size=config.V,
            max_context_size=config.max_context,
            num_encoders=config.N,
            num_decoders=config.N,
            n_heads=config.h,
            droprate=config.drop_rate,
            position_encoding_class=config.position_encoding_class,
            verbose=config.verbose,
        )

    def forward_encoders(self, enc_input):
        """Forward fn for encoders only."""
        enc = self.embedding(enc_input)
        enc = self.position_encoder(enc)
        enc = self.emb_drop(enc)
        return self.encoders(enc)

    def forward_decoders(self, dec_input, enc):
        """Forward fn for decoders only."""
        dec = self.embedding(dec_input)
        dec = self.position_encoder(dec)
        dec = self.emb_drop(dec)

        for decoder in self.decoders:
            dec = decoder(dec, enc)

        return dec

    def forward(self, enc_input, dec_input):
        if self.cache_encoder_output:
            enc = self.prev_enc
            if enc is None:
                enc = self.forward_encoders(enc_input)
                self.prev_enc = enc
        else:
            enc = self.forward_encoders(enc_input)

        dec = self.forward_decoders(dec_input, enc)

        logits = self.linear(dec)
        return logits

    def reset_cache(self):
        self.cache_encoder_output = True
        self.prev_enc = None


class Autoregressive(nn.Module):
    """Simple implementation of an autoregressive model using a stack of decoders only."""

    def __init__(
        self,
        emb_dimension: int,
        attn_dimension: int,
        vocab_size: int,
        max_context_size: int,
        num_decoders: int,
        n_heads: int,
        droprate: float = 0.1,
        position_encoding_class: Type[nn.Module] = HardcodedPositionalEncoding,
        verbose: bool = False,
    ):
        assert (
            attn_dimension % n_heads == 0
        ), f"attn_dimension must be divisible by n_head. Got {attn_dimension} and {n_heads}."
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_dimension)
        self.position_encoder = position_encoding_class(
            emb_dimension,
            max_context=max_context_size,
        )
        self.emb_drop = nn.Dropout(droprate)

        self.decoders = nn.Sequential(
            *[
                AutoregressiveDecoder(
                    n_heads,
                    attn_dimension,
                    max_context_size,
                    droprate=droprate,
                    verbose=verbose,
                )
                for _ in range(num_decoders)
            ]
        )

        self.linear = nn.Linear(attn_dimension, vocab_size)

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(
            emb_dimension=config.dim,
            attn_dimension=config.dim,
            vocab_size=config.V,
            max_context_size=config.max_context,
            num_decoders=config.N,
            n_heads=config.h,
            droprate=config.drop_rate,
            position_encoding_class=config.position_encoding_class,
            verbose=config.verbose,
        )

    def forward(self, X):
        emb = self.embedding(X)
        emb = self.position_encoder(emb)
        emb = self.emb_drop(emb)

        dec = self.decoders(emb)

        logits = self.linear(dec)
        return logits
