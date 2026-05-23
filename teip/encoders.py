import torch
import torch.nn as nn

class CharEncoder(nn.Module):
    def __init__(self, vocab_size: int = 21, d_model: int = 128, max_len: int = 34, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(max_len*d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.ReLU())
    def forward(self, tokens):
        return self.encoder(self.embedding(tokens))

class CrossFusion(nn.Module):
    def __init__(self, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d*2, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d), nn.ReLU())
    def forward(self, peptide_vec, hla_vec):
        return self.net(torch.cat([peptide_vec, hla_vec], dim=-1))
