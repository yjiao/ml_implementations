"""
Functions related to relative position representations from Shaw et al 2018.

Separated from torch_models.py since most functions have to be re-implemented.
Relative positional encodings are used within each attention fn and are not
drop-in replacements for learned absolute positional encodings.

note that throughout rpe = Relative Positional Encoding

Tests out a few different configurations:
    - global RPE (more similar to absolute positional encoding wrt # params)
    - per-layer RPE
    - per-layer absolute positional embedding

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial
import logging

import math
import numpy as np

from collections import OrderedDict

from typing import Callable, Optional, Type

from . import torch_models as tm


@dataclass
class RPEModelConfig:
    dim: int  # dimension of embeddings and encoder/ decoder outputs
    N: int  # number of encoders and decoders
    h: int  # number of heads
    V: int  # vocab size
    max_context: int

    # if true, use RPE. else use learnable positional embedding
    relative: bool

    # whether each layer should own its own PE
    # if False, global PEs are owned by the model and passed to submodules
    per_layer: bool

    # if True, both key and value RPE are used
    # else only key RPE is used
    use_value_rpe: bool

    # max distance between tokens beyond which distances are clipped
    max_distance: int

    drop_rate: float = 0.1
    verbose: bool = False


class RelativePositionalEncodingFast(nn.Module):
    """Implementation of RPE as described in the music transformer paper (Huang et al 2018).

    Note that this only works for causal self attention, which assumes that we
    want a square matrix output. A different padding scheme is required for
    rectangular outputs."""

    def __init__(self, max_distance: int, dimension: int, *args, **kwargs):
        super().__init__()

        self.max_distance = max_distance
        emb = torch.randn(max_distance, dimension)

        self.embeddings = nn.Parameter(emb)

    def forward(self, Q: torch.tensor):
        """Apply rpe skew operation."""
        # Q: (batch, heads, qlen, dim)
        batch, h, qlen, d = Q.shape
        srel = self.embeddings[-qlen:, :]
        if qlen > self.max_distance:
            # distance
            # -2 [x x x ...]
            # -1 [x x x ...]
            #  0 [x x x ...]
            # qlen = 5
            # -2 [x x x ...]
            # -2 [x x x ...]
            # -2 [x x x ...]
            # -1 [x x x ...]
            #  0 [x x x ...]
            clip_emb = srel[0].repeat(qlen - self.max_distance, 1)
            srel = torch.cat((clip_emb, srel), 0)
        srel = Q @ srel.T
        srel = F.pad(srel, [1, 0])
        srel = srel.view((batch, h, qlen + 1, qlen))
        return srel[:, :, -qlen:, :qlen]


class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_context_size: int, max_distance: int, dimension: int):
        super().__init__()
        dist_mat = self._make_distance_mat(max_context_size, max_distance)
        self.register_buffer("dist_mat", dist_mat)

        vocab_sz = 2 * max_distance + 1
        emb = torch.randn(vocab_sz, dimension)
        self.embeddings = nn.Parameter(emb)

    def _make_distance_mat(self, max_context: int, max_distance: int):
        """Used for indexing into relative position embedding matrix.

        Return positive relative idx for indexing into self.embeddings.

        Ex. i = 1, seqlen = 7, max_distance = 3
           idx of query:    1
            idx of keys: 0  1  2  3  4  5  6
        rel. position: [-1, 0, 1, 2, 3, 4, 5]
            clip >= 3: [-1, 0, 1, 2, 3, 3, 3]
        make positive: [ 2, 3, 4, 5, 6, 6, 6]
        """
        # create all relative positions to be used for querying self.embeddings
        dist_mat = np.tile(np.arange(max_context), (max_context, 1))
        dist_mat -= np.transpose(dist_mat)
        dist_mat = np.clip(dist_mat, -max_distance, max_distance)
        dist_mat += max_distance
        return torch.tensor(dist_mat)

    def forward(self, ilen, jlen):
        """get a_ij matrix from self.embeddings"""
        idx = self.dist_mat[:ilen, :jlen]
        idx_flat = idx.contiguous().view(ilen * jlen).to(self.embeddings.device)
        emb_flat = torch.index_select(self.embeddings, 0, idx_flat).to(
            self.embeddings.device
        )
        return emb_flat.view(ilen, jlen, -1)


class RelativeCausalAttention(tm.SelfAttentionMulti):
    """
    Performs RELATIVE multi-headed self attention with masking.

    Does not implement the feed forward layers after attention.

    Input dimension == output dimension.
    """

    def __init__(
        self,
        n_heads: int,
        dim_io: int,
        max_context_size: int,
        key_rpe: nn.Module,
        val_rpe: Optional[nn.Module],  # relative position embeddings
        causal: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            causal=causal,
            verbose=verbose,
        )
        self.key_rpe = key_rpe
        self.val_rpe = val_rpe

    def apply_rpe(self, x, rpe, transpose: bool):
        """Apply RPE for attention.
        2 scenarios:
            x = Q, rpe = Key RPE
                x: (batch, n_heads, qlen, dim_per_head)
                rpe: (qlen, klen, dim_per_head)
            x = alpha, rpe = Value RPE
                alpha: (batch, n_heads, qlen, klen)
                rpe: (qlen, klen, dim_per_head)
        """
        batch, n_heads, qlen, d_or_klen = x.shape
        # (qlen, n_heads, batch, dim_per_head)
        # - or -
        # (qlen, n_heads, batch, klen)
        x_t = x.permute(2, 0, 1, 3)
        x_t = x_t.contiguous().view((qlen, batch * n_heads, d_or_klen))

        if transpose:
            # (qlen, dim_per_head, klen)
            rpe = rpe.transpose(1, 2)

        # scenario 1 (transpose = True)
        #   (qlen, batch * n_heads, dim_per_head)
        # x (qlen, dim_per_head, klen)
        # ->(qlen, batch * n_heads, klen)

        # scenario 2 (transpose = False)
        #   (qlen, batch * n_heads, klen)
        # x (qlen, klen, dim_per_head)
        # ->(qlen, batch * n_heads, dim_per_head)
        rpe = torch.matmul(x_t, rpe)
        rpe = rpe.view((qlen, batch, n_heads, -1))
        # (batch, n_heads, qlen, dim_per_head or klen)
        rpe = rpe.permute(1, 2, 0, 3)
        return rpe

    def attention(
        self, k: torch.tensor, q: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        # k, q, v have dim (batch, n_heads, seqlen, dim_per_head)
        qk = torch.matmul(q, k.transpose(-1, -2))

        # reshape for relative position embeddings
        batch, n_heads, qlen, dim_per_head = q.shape
        _, _, klen, _ = k.shape

        # add relative positional embeddings
        # (qlen, klen, dim_per_head)
        k_rpe = self.key_rpe(qlen, klen)

        # (qlen, n_heads, batch, dim_per_head)
        logits = qk + self.apply_rpe(q, k_rpe, transpose=True)
        logits /= np.sqrt(k.shape[-1])

        if self.causal:
            logits.masked_fill_(self.mask[:qlen, :klen] == 0, -1e9)

        # (batch, n_heads, qlen, klen)
        weights = nn.functional.softmax(logits, dim=-1)
        # (batch, n_heads, qlen, klen) x (batch, n_heads, klen, dim_per_head)
        # (batch, n_heads, qlen, dim_per_head)
        qkv = torch.matmul(weights, v)
        if self.val_rpe is not None:
            v_rpe = self.val_rpe(qlen, klen)
            qkv += self.apply_rpe(weights, v_rpe, transpose=False)
        return qkv

    def forward(self, X: torch.tensor) -> torch.tensor:
        # X: (batch, seqlen, dim_io)
        # W: (dim_io, dim_io)
        # dim_io = dim_per_head * n_heads
        k = self.Wk(X)  # output shape: (batch, seqlen, dim_io)
        q = self.Wq(X)
        v = self.Wv(X)

        attn = self.multihead_attention(k, q, v)
        return attn


class RelativeCausalAttentionFast(RelativeCausalAttention):
    """
    Performs RELATIVE multi-headed self attention with masking using Music Transformer skewing.

    Does not implement the feed forward layers after attention.

    Input dimension == output dimension.
    """

    def attention(
        self, k: torch.tensor, q: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        # k, q, v have dim (batch, n_heads, seqlen, dim_per_head)
        qk = torch.matmul(q, k.transpose(-1, -2))

        # reshape for relative position embeddings
        batch, n_heads, qlen, dim_per_head = q.shape
        _, _, klen, _ = k.shape

        # add relative positional embeddings
        # (qlen, klen, dim_per_head)
        k_rpe = self.key_rpe(q)

        # (qlen, n_heads, batch, dim_per_head)
        logits = qk + k_rpe
        logits /= np.sqrt(k.shape[-1])

        if self.causal:
            logits.masked_fill_(self.mask[:qlen, :klen] == 0, -1e9)

        # (batch, n_heads, qlen, klen)
        weights = nn.functional.softmax(logits, dim=-1)
        # (batch, n_heads, qlen, klen) x (batch, n_heads, klen, dim_per_head)
        # (batch, n_heads, qlen, dim_per_head)
        qkv = torch.matmul(weights, v)
        if self.val_rpe is not None:
            v_rpe = self.val_rpe(q)
            qkv += v_rpe
        return qkv


class RelativeDecoder(tm.AutoregressiveDecoder):
    """RelativeDecoder module with causal self-attention.

    Main difference from parent class is that self.causal_attn is replaced with
    RelativeCausalAttention.
    """

    def __init__(
        self,
        n_heads: int,
        dim_io: int,
        max_context_size: int,
        per_layer: bool,
        use_value_rpe: bool,
        key_rpe: Optional[nn.Module] = None,
        val_rpe: Optional[nn.Module] = None,
        pe_ctor: Optional[Callable] = None,
        droprate: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            droprate=droprate,
            verbose=verbose,
        )

        # dimension in must = dimension out
        assert dim_io % n_heads == 0, (
            "input dimension must be divisible by the number of heads. "
            + f"Got {dim_io} input dimension and {n_heads} heads."
        )

        if per_layer:
            assert (
                pe_ctor is not None
            ), "Constructor for RPE must be supplied when per_layer = True."
            if key_rpe is not None or val_rpe is not None:
                logging.warn(
                    "At least one of key_rpe or val_rpe was supplied but per_layer is True. key_rpe and val_rpe will be ignored."
                )
        else:
            assert (
                key_rpe is not None
            ), "When per_layer is False, key_rpe must be passed as an argument."

        if per_layer:
            key_rpe = pe_ctor()
            if use_value_rpe:
                val_rpe = pe_ctor()
            else:
                val_rpe = None

        # overwrite parent causal_attn attribute
        self.causal_attn = RelativeCausalAttentionFast(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            key_rpe=key_rpe,
            val_rpe=val_rpe,
            causal=True,
            verbose=verbose,
        )


class PEDecoder(tm.AutoregressiveDecoder):
    """Decoder module with causal self-attention, with PER-LAYER PE.

    Main difference with parent class is the presence of LearnablePositionalEncoding."""

    def __init__(
        self,
        n_heads: int,
        dim_io: int,
        max_context_size: int,
        per_layer: bool,
        pe: Optional[nn.Module] = None,
        pe_ctor: Optional[Callable] = None,
        droprate: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(
            n_heads=n_heads,
            dim_io=dim_io,
            max_context_size=max_context_size,
            droprate=droprate,
            verbose=verbose,
        )
        if per_layer:
            assert (
                pe_ctor is not None
            ), "Constructor for RPE must be supplied when per_layer = True."
            self.pe = pe_ctor()
        else:
            assert pe is not None, "pe must be supplied when per_layer is False."
            self.pe = pe

    def forward(self, X: torch.tensor) -> torch.tensor:
        dec = self.pe(X)
        return super().forward(self.pe(X))


class RelativeGPT(nn.Module):
    """GPT-style decoder stack with various positional embedding configs:

    - RPE as described in Shaw et al 2018
    - learnable PE

    Both options can be either global (used by all attention layers) or per-layer.
    Note thatt this is usually not the way learnable PEs are used, but rather are implemented
    this way to provide a direct comparison with RPE.
    """

    def __init__(
        self,
        emb_dimension: int,
        attn_dimension: int,
        vocab_size: int,
        max_context_size: int,
        num_decoders: int,
        n_heads: int,
        max_distance: int,
        use_value_rpe: bool,
        relative: bool,
        per_layer: bool,
        droprate: float = 0.1,
        verbose: bool = False,
    ):
        assert (
            attn_dimension % n_heads == 0
        ), f"attn_dimension must be divisible by n_head. Got {attn_dimension} and {n_heads}."
        super().__init__()
        self.embedding = tm.Embedding(vocab_size, emb_dimension)
        self.emb_drop = nn.Dropout(droprate)

        # because there are many possible configs, we create two functools.partial functions
        # that will eventually be called by submodules with no arguments
        # - pe_ctor: constructor for RPE or PE
        # - decoder_ctor: constructor for decoder, either with PE or RPE
        if relative:  # RPE
            pe_ctor = partial(
                RelativePositionalEncodingFast,
                max_context_size=max_context_size,
                max_distance=max_distance,
                dimension=int(attn_dimension / n_heads),
            )
            decoder_ctor = partial(
                RelativeDecoder,
                n_heads=n_heads,
                dim_io=attn_dimension,
                max_context_size=max_context_size,
                per_layer=per_layer,
                use_value_rpe=use_value_rpe,
                droprate=droprate,
                verbose=verbose,
            )
        else:
            pe_ctor = partial(
                tm.LearnablePositionalEncoding,
                max_context=max_context_size,
                d_model=attn_dimension,
            )
            decoder_ctor = partial(
                PEDecoder,
                n_heads=n_heads,
                dim_io=attn_dimension,
                max_context_size=max_context_size,
                per_layer=per_layer,
                droprate=droprate,
                verbose=verbose,
            )

        if not per_layer:
            if relative:
                # make global rpes used by every layer in the model
                decoder_ctor = partial(
                    decoder_ctor,
                    key_rpe=pe_ctor(),
                    val_rpe=pe_ctor() if use_value_rpe else None,
                )
            else:
                decoder_ctor = partial(decoder_ctor, pe=pe_ctor())
        else:
            decoder_ctor = partial(decoder_ctor, pe_ctor=pe_ctor)

        self.decoders = nn.Sequential(*[decoder_ctor() for _ in range(num_decoders)])

        self.linear = nn.Linear(attn_dimension, vocab_size)

    @classmethod
    def from_config(cls, config: RPEModelConfig):
        return cls(
            emb_dimension=config.dim,
            attn_dimension=config.dim,
            vocab_size=config.V,
            max_context_size=config.max_context,
            num_decoders=config.N,
            n_heads=config.h,
            max_distance=config.max_distance,
            droprate=config.drop_rate,
            verbose=config.verbose,
            use_value_rpe=config.use_value_rpe,
            relative=config.relative,
            per_layer=config.per_layer,
        )

    def forward(self, X):
        emb = self.embedding(X)
        emb = self.emb_drop(emb)
        dec = self.decoders(emb)
        logits = self.linear(dec)
        return logits
