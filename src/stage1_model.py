
"""
Stage 1 Model: BiLSTM + Additive Attention + Dual Heads
  - Classification head: 28-emotion multi-label
  - Regression head: intensity score [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over BiLSTM outputs.
    Learns which words matter most for emotion prediction.
    Returns context vector and per-token attention weights.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
            mask:     (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            context:  (batch, hidden_dim) — weighted sum of LSTM outputs
            weights:  (batch, seq_len)    — attention weight per token
        """
        energy = torch.tanh(self.W(lstm_out))           # (B, L, H)
        scores = self.v(energy).squeeze(-1)              # (B, L)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)               # (B, L)
        context = torch.bmm(weights.unsqueeze(1),
                            lstm_out).squeeze(1)          # (B, H)
        return context, weights


class EmotionDetector(nn.Module):
    """
    Complete Stage 1 architecture:

    Input token IDs
         ↓
    Embedding Layer (vocab_size × embed_dim)
         ↓
    Bidirectional LSTM (num_layers, hidden_dim)
         ↓
    Additive Attention → context vector
         ↓          ↓
    Classifier    Intensity Head
    (28 logits)   (sigmoid → scalar)
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_classes: int, num_layers: int, dropout: float,
                 pad_idx: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        bidir_dim = hidden_dim * 2  # because bidirectional

        self.attention = AdditiveAttention(bidir_dim)
        self.dropout = nn.Dropout(dropout)

        # Classification head: multi-label (28 emotions)
        self.classifier = nn.Sequential(
            nn.Linear(bidir_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Intensity regression head: outputs scalar in [0, 1]
        self.intensity_head = nn.Sequential(
            nn.Linear(bidir_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            input_ids: (batch, seq_len)  — padded token indices
            lengths:   (batch,)          — actual sequence lengths
        Returns:
            logits:      (batch, num_classes) — raw scores (before sigmoid)
            intensity:   (batch,)             — predicted intensity in [0,1]
            attn_weights:(batch, seq_len)     — attention weights per token
        """
        batch_size, seq_len = input_ids.size()

        # Mask: 1 for real tokens, 0 for padding
        mask = (input_ids != 0).float()

        # Embedding + dropout
        embedded = self.dropout(self.embedding(input_ids))  # (B, L, E)

        # Pack → LSTM → Unpack
        packed = pack_padded_sequence(
            embedded, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )  # (B, L, 2H)

        lstm_out = self.dropout(lstm_out)

        # Attention → context vector
        context, attn_weights = self.attention(lstm_out, mask)

        # Dual heads
        logits = self.classifier(context)                        # (B, C)
        intensity = self.intensity_head(context).squeeze(-1)     # (B,)

        return logits, intensity, attn_weights
