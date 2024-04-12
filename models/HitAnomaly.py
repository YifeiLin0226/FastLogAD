import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, dff, heads):
        super(TransformerBlock, self).__init__()
        self.query_weight = nn.Linear(hidden_dim, hidden_dim)
        self.key_weight = nn.Linear(hidden_dim, hidden_dim)
        self.value_weight = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, hidden_dim),
        )
    def forward(self, x):
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)
        x, _ = self.attention(query, key, value)
        x = self.feed_forward(x)
        return x



class LogEncoder(nn.Module):
    def __init__(self, hidden_dim, dff, heads):
        super(LogEncoder, self).__init__()
        self.transformer_block1 = TransformerBlock(hidden_dim, dff, heads)
        self.transformer_block2 = TransformerBlock(hidden_dim, dff, heads)

        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = x.transpose(-1, -2)
        x = self.pooling(x).squeeze(-1)
        return x

class LogSequenceEncoder(nn.Module):
    def __init__(self, hidden_dim, diff, heads):
        super(LogSequenceEncoder, self).__init__()
        self.log_encoder = LogEncoder(hidden_dim, diff, heads)
        self.transformer_block = TransformerBlock(hidden_dim, diff, heads)
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.log_encoder(x)
        x = self.transformer_block(x)
        x = x.transpose(-1, -2)
        x = self.pooling(x).squeeze(-1)
        return x

class ParamEncoder(nn.Module):
    def __init__(self, hidden_dim, dff, heads):
        super(ParamEncoder, self).__init__()
        self.para_encoder = LogEncoder(hidden_dim, dff, heads)
        self.para_pooling = nn.AdaptiveAvgPool1d(1)
        self.log_encoder = LogEncoder(hidden_dim, dff, heads)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.weight1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.weight2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, logseq, param):
        param = self.para_encoder(param)
        param = param.transpose(-1, -2)
        param = self.para_pooling(param).squeeze(-1)
        logseq = self.log_encoder(logseq)

        out = self.weight1(param) + self.weight2(logseq) + self.b
        out = self.pooling(out.transpose(-1, -2)).squeeze(-1)
        return out


class HitAnomaly(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dff, heads):
        super(HitAnomaly, self).__init__()
        self.project_embedding1 = nn.Linear(embed_dim, hidden_dim)
        self.project_embedding2 = nn.Linear(embed_dim, hidden_dim)
        self.logseq_encoder = LogSequenceEncoder(hidden_dim, dff, heads)
        self.param_encoder = ParamEncoder(hidden_dim, dff, heads)

        self.W_alpha = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.v = nn.Parameter(torch.randn(hidden_dim, 1))
    
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, logseq, param):
        logseq = self.project_embedding1(logseq)
        param = self.project_embedding2(param)
        Rp = self.param_encoder(logseq, param)
        Rs = self.logseq_encoder(logseq)
        fRs = torch.tanh(Rs @ self.W_alpha) @ self.v
        fRp = torch.tanh(Rp @ self.W_alpha) @ self.v
        alpha_s = torch.exp(fRs) / (torch.exp(fRs) + torch.exp(fRp))
        alpha_p = 1 - alpha_s
        out = alpha_s * Rs + alpha_p * Rp
        out = self.linear(out)
        return out