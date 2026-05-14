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


def sinusoidal_pe(max_len: int, d_model:int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float() # (max_len, 1)
    div = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10_000.0) / d_model)
    ) # (d_model/2,)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe # (max_len, d_model)


class EmbeddingBlock(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.tok_emb = nn.Embedding(
            config.vocab_size, 
            config.d_model,
            padding_idx=config.pad_id
        )
        self.register_buffer("pos_emb", sinusoidal_pe(config.max_len, config.d_model))
        self.dropout  = nn.Dropout(config.dropout)

    def forward(self, input_ids):          # (B, L)
        x = self.tok_emb(input_ids)                   # (B, L, D)  learned
        x = x + self.pos_emb[:input_ids.size(1)]      # (B, L, D)  fixed
        return self.dropout(x)
    

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_head
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.d_model, 3*config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.resid_drop = nn.Dropout(config.dropout)
    
    def forward(self, x, pad_mask=None):
        B, L, D = x.shape   #B: batch size, L: seq length, D: model dim

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)  # (B, L, D) each

        def split_heads(t):
            return t.view(B, L, self.n_head, self.d_head).transpose(1, 2)        
        q, k, v = split_heads(q), split_heads(k), split_heads(v)   # (B, n_head, L, d_head) each

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, n_head, L, L)

        if pad_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device)
            attn_mask = attn_mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)                    # (B, n_head, L, L) clean — returned
        attn_drop    = F.dropout(attn_weights, p=self.dropout, training=self.training)
        y            = attn_drop @ v                                 # (B, n_head, L, d_head)

        y = y.transpose(1, 2).contiguous().view(B, L, D)            # (B, L, D)
        y = self.resid_drop(self.c_proj(y))

        return y, attn_weights  # attn_weights is pre-dropout, used for Q10-12


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.ff   = nn.Sequential(
            nn.Linear(config.d_model, 4*config.d_model),
            nn.GELU(),
            nn.Linear(4*config.d_model, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x, pad_mask=None):
        attn_out, attn_w = self.attn(x, pad_mask)
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
        
        self.detect_head = nn.Linear(config.d_model, 2)
        self.correct_head = nn.Linear(config.d_model, 10)
    
    def forward(self, input_ids, pad_mask, return_internals=False):
        x = self.embed(input_ids)

        hidden_states = []
        attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x, pad_mask)
            if return_internals:
                hidden_states.append(x)
                attn_weights.append(attn_w)
        
        x = self.ln_f(x)

        detect_logits = self.detect_head(x[:, 0, :]) # (B, L, 2)
        correct_logits = self.correct_head(x) # (B, L, 10)

        if return_internals:
            return detect_logits, correct_logits, hidden_states, attn_weights
        else:
            return detect_logits, correct_logits