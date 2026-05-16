import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 7
    n_layer: int    = 4
    n_head: int     = 4
    d_model: int    = 128
    dropout: float  = 0.1
    max_len: int    = 80
    pad_id: int     = 0
    # ── Q18: relative positional encoding (T5-style) ──────────────────────
    use_rel_pos: bool      = False  # if True, disables absolute PE and uses RPE
    rel_pos_buckets: int   = 32     # number of buckets for the relative-position table
    rel_pos_max_dist: int  = 128    # distances beyond this are clipped to the last bucket


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10_000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class EmbeddingBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_rel_pos = config.use_rel_pos
        self.tok_emb = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        if not config.use_rel_pos:
            self.register_buffer(
                "pos_emb", sinusoidal_pe(config.max_len, config.d_model)
            )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        if not self.use_rel_pos:
            x = x + self.pos_emb[: input_ids.size(1)]
        return self.dropout(x)


# ── Q18: T5-style relative position bias ─────────────────────────────────────


def _relative_position_bucket(
    relative_position: torch.Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Map relative positions to bucket indices (T5 convention).

    Half the buckets cover offsets in [-max_distance/2, max_distance/2] linearly;
    the other half cover larger absolute offsets on a logarithmic scale.
    Sign is preserved by splitting the table into negative- and positive-offset halves.
    """
    ret = 0
    n = -relative_position
    num_buckets //= 2
    ret += (n < 0).to(torch.long) * num_buckets
    n = torch.abs(n)

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.minimum(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )

    ret += torch.where(is_small, n, val_if_large)
    return ret


class RelativePositionBias(nn.Module):
    """Learned per-head bias indexed by bucketed relative position.
    Shared across layers (T5 convention) for parameter efficiency."""

    def __init__(self, n_head: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.n_head       = n_head
        self.num_buckets  = num_buckets
        self.max_distance = max_distance
        self.bias = nn.Embedding(num_buckets, n_head)

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(L, device=device)[:, None]
        k_pos = torch.arange(L, device=device)[None, :]
        rel_pos = k_pos - q_pos
        buckets = _relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.bias(buckets)                  # (L, L, n_head)
        return values.permute(2, 0, 1).contiguous()  # (n_head, L, L)


# ── Attention ────────────────────────────────────────────────────────────────


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.n_head  = config.n_head
        self.d_model = config.d_model
        self.d_head  = config.d_model // config.n_head
        self.dropout = config.dropout

        self.c_attn     = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj     = nn.Linear(config.d_model, config.d_model)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, pad_mask=None, rel_pos_bias=None):
        """`rel_pos_bias` is an optional (n_head, L, L) tensor added to scores."""
        B, L, D = x.shape

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        def split_heads(t):
            return t.view(B, L, self.n_head, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias.unsqueeze(0)   # (1, H, L, L) broadcast

        if pad_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device)
            attn_mask = attn_mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_drop    = F.dropout(attn_weights, p=self.dropout, training=self.training)
        y            = attn_drop @ v

        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.resid_drop(self.c_proj(y))

        return y, attn_weights


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x, pad_mask=None, rel_pos_bias=None):
        attn_out, attn_w = self.attn(x, pad_mask, rel_pos_bias=rel_pos_bias)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x, attn_w


class DyckTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed  = EmbeddingBlock(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f   = nn.LayerNorm(config.d_model)

        if config.use_rel_pos:
            self.rel_pos = RelativePositionBias(
                n_head=config.n_head,
                num_buckets=config.rel_pos_buckets,
                max_distance=config.rel_pos_max_dist,
            )
        else:
            self.rel_pos = None

        self.detect_head  = nn.Linear(config.d_model, 2)
        self.correct_head = nn.Linear(config.d_model, 10)

    def forward(self, input_ids, pad_mask, return_internals=False):
        B, L = input_ids.shape
        x = self.embed(input_ids)

        rel_pos_bias = (
            self.rel_pos(L, input_ids.device) if self.rel_pos is not None else None
        )

        hidden_states = []
        attn_weights  = []
        for block in self.blocks:
            x, attn_w = block(x, pad_mask, rel_pos_bias=rel_pos_bias)
            if return_internals:
                hidden_states.append(x)
                attn_weights.append(attn_w)

        x = self.ln_f(x)

        detect_logits  = self.detect_head(x[:, 0, :])
        correct_logits = self.correct_head(x)

        if return_internals:
            return detect_logits, correct_logits, hidden_states, attn_weights
        else:
            return detect_logits, correct_logits
