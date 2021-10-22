import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import math

def clones(module, N) -> nn.ModuleList:
    "clone module for N times"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Scaled Dot Product Attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, d_q, d_k, d_v, d_model, dropout=0.2):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head == 0
        self.linears = nn.ModuleList([
            nn.Linear(d_q, d_model),
            nn.Linear(d_k, d_model),
            nn.Linear(d_v, d_model),
            nn.Linear(d_model, d_model)
        ])
        self.d_k = d_model // head
        self.head = head
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, src_len)
        batch = query.size(0)
        query, key, value = \
            (l(x).view(batch, -1, self.head, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value)))
        x, self.attn = attention(query, key, value, mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.head * self.d_k)
        return self.linears[-1](x)
