import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.k = nn.Linear(32, 32)
        # self.q = nn.Linear(32, 32)
        # self.v = nn.Linear(32, 32)

    def forward(self, q, k, v, mask=None):
        # q = self.q(q)
        # k = self.k(k)
        # v = self.v(v)
        # attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = torch.matmul(q / self.temperature, k.transpose(0, 1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
