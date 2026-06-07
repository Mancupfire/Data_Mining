"""
SASRec: Self-Attentive Sequential Recommendation
Kang & McAuley, ICDM 2018
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn  = PointWiseFeedForward(hidden_dim, dropout)
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # Self-attention with causal mask
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, is_causal=False)
        x = self.drop(x) + residual

        # Feed-forward
        residual = x
        x = self.ln2(x)
        x = self.drop(self.ffn(x)) + residual
        return x


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, max_len=50,
                 num_heads=1, num_blocks=2, dropout=0.2):
        super().__init__()
        self.num_items  = num_items
        self.hidden_dim = hidden_dim
        self.max_len    = max_len

        # item embedding: index 0 = padding
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, hidden_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seq):
        """
        seq: (B, L) item indices, 0 = padding
        returns: (B, L, hidden_dim) sequence representations
        """
        B, L = seq.shape
        device = seq.device

        positions = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_drop(x)

        # Causal (upper-triangular) attention mask — shape (L, L)
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
        )
        # Also mask out padding positions
        pad_mask = (seq == 0)  # (B, L)

        # PyTorch MHA expects float mask; True → -inf
        attn_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln(x)
        return x  # (B, L, D)

    def encode(self, seq):
        """Encode sequence and return last-position representation."""
        h = self.forward(seq)                                     # (B, L, D)
        lengths = (seq != 0).sum(dim=1) - 1                      # (B,)
        lengths = lengths.clamp(min=0)
        return h[torch.arange(len(seq)), lengths]                 # (B, D)

    def predict(self, seq, item_indices):
        """
        seq: (B, L)
        item_indices: (B, K) — items to score
        returns: (B, K) logits
        """
        h_last = self.encode(seq)                                 # (B, D)
        item_embs = self.item_emb(item_indices)                   # (B, K, D)
        logits = torch.bmm(item_embs, h_last.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return logits

    def score_all(self, seq, all_item_embs):
        """
        seq: (B, L)
        all_item_embs: (N, D) — precomputed embeddings for all items
        returns: (B, N) scores
        """
        h_last = self.encode(seq)   # (B, D)
        return h_last @ all_item_embs.T  # (B, N)
