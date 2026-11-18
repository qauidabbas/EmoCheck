
"""
Dataset and DataLoader utilities for Stage 1 emotion detection.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from .config import NUM_EMOTIONS, IDX_TO_LABEL, EMOTION_BASE_INTENSITY
from .preprocessing import Vocabulary


def compute_intensity_proxy(label_indices: List[int]) -> float:
    """
    Heuristic intensity proxy.
    GoEmotions has no intensity labels, so we approximate:
      - Use base intensity per emotion type
      - Add complexity bonus for multi-label samples
      - Add small random noise for regularization
    """
    if not label_indices:
        return 0.1

    emotions = [IDX_TO_LABEL[i] for i in label_indices if i < NUM_EMOTIONS]
    if not emotions:
        return 0.1

    base_vals = [EMOTION_BASE_INTENSITY.get(e, 0.4) for e in emotions]
    mean_intensity = np.mean(base_vals)

    # More co-occurring emotions → higher complexity
    complexity_bonus = min(0.15 * (len(emotions) - 1), 0.3)
    intensity = mean_intensity + complexity_bonus

    # Small noise for regularization
    noise = random.uniform(-0.05, 0.05)
    return float(np.clip(intensity + noise, 0.05, 0.98))


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for Stage 1.
    Each item: (token_ids, length, multi_hot_labels, intensity)
    """
    def __init__(self, texts: List[str], labels: List[List[int]],
                 vocab: Vocabulary, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels[idx]

        # Numericalize and truncate
        ids = self.vocab.numericalize(text)[:self.max_len]
        length = max(len(ids), 1)

        # Multi-hot label vector (28 emotions)
        label_vec = torch.zeros(NUM_EMOTIONS, dtype=torch.float)
        for li in label_indices:
            if li < NUM_EMOTIONS:
                label_vec[li] = 1.0

        intensity = compute_intensity_proxy(label_indices)

        return {
            "text": text,
            "ids": torch.tensor(ids, dtype=torch.long),
            "length": length,
            "labels": label_vec,
            "intensity": torch.tensor(intensity, dtype=torch.float),
        }


def collate_emotion(batch):
    """
    Custom collate function to pad variable-length sequences.
    """
    ids_list = [item["ids"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.stack([item["labels"] for item in batch])
    intensities = torch.stack([item["intensity"] for item in batch])
    texts = [item["text"] for item in batch]

    # Pad all sequences to the length of the longest in this batch
    ids_padded = pad_sequence(ids_list, batch_first=True, padding_value=0)

    # Safety: clamp to max 128
    max_len = min(ids_padded.size(1), 128)
    ids_padded = ids_padded[:, :max_len]
    lengths = lengths.clamp(max=max_len)

    return {
        "texts": texts,
        "ids": ids_padded,
        "lengths": lengths,
        "labels": labels,
        "intensities": intensities,
    }
