"""
GRU4Rec: Session-based Recommendations with Recurrent Neural Networks
Hidasi et al., ICLR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=100, max_len=50,
                 num_layers=1, dropout=0.2):
        super().__init__()
        self.num_items  = num_items
        self.hidden_dim = hidden_dim
        self.max_len    = max_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seq):
        """
        seq: (B, L) item indices
        returns: (B, L, hidden_dim)
        """
        x = self.drop(self.item_emb(seq))   # (B, L, D)
        out, _ = self.gru(x)                # (B, L, D)
        out = self.drop(out)
        return out  # (B, L, D)

    def encode(self, seq):
        """Encode sequence and return last-position representation."""
        h = self.forward(seq)
        lengths = (seq != 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        h_last = h[torch.arange(len(seq)), lengths]
        return self.out(h_last)   # (B, D)

    def predict(self, seq, item_indices):
        """
        seq: (B, L)
        item_indices: (B, K)
        returns: (B, K) logits
        """
        h_last = self.encode(seq)                              # (B, D)
        item_embs = self.item_emb(item_indices)               # (B, K, D)
        logits = torch.bmm(item_embs, h_last.unsqueeze(-1)).squeeze(-1)
        return logits

    def score_all(self, seq, all_item_embs):
        """
        seq: (B, L)
        all_item_embs: (N, D)
        returns: (B, N) scores
        """
        h_last = self.encode(seq)
        return h_last @ all_item_embs.T
