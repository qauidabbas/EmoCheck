
"""
Stage 1 Inference: takes raw text, returns full structured emotion output.
"""
import torch
import numpy as np
from typing import Dict, List, Any

from .config import NUM_EMOTIONS, EMOTION_LABELS
from .preprocessing import Vocabulary
from .stage1_model import EmotionDetector


class Stage1Predictor:
    """
    Runs the trained EmotionDetector on raw text.
    Returns structured output with all 28 emotion probabilities,
    top-k emotions, intensity score, and attention weights.
    """
    def __init__(self, model: EmotionDetector, vocab: Vocabulary,
                 max_len: int = 128, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab = vocab
        self.max_len = max_len

    def predict(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Full inference for a single text string.

        Returns:
            {
                "text": "...",
                "top_emotions": [{"emotion": "sadness", "confidence": 0.55}, ...],
                "all_emotion_scores": {"admiration": 0.01, ...},
                "intensity": 0.52,
                "attention_weights": [{"word": "empty", "weight": 0.23}, ...],
                "raw_probs": np.array(...)   # for Stage 2 input
            }
        """
        tokens = self.vocab.tokens_from_text(text)
        ids = self.vocab.numericalize(text)[:self.max_len]
        length = max(len(ids), 1)

        ids_tensor = torch.tensor([ids], dtype=torch.long).to(self.device)
        len_tensor = torch.tensor([length], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits, intensity, attn_weights = self.model(ids_tensor, len_tensor)

        # Sigmoid probabilities (multi-label)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        intensity_val = intensity.item()
        attn = attn_weights.squeeze(0).cpu().numpy()[:length]

        # All 28 emotion scores
        all_scores = {
            EMOTION_LABELS[i]: round(float(probs[i]), 4)
            for i in range(NUM_EMOTIONS)
        }

        # Top-k emotions sorted by probability
        sorted_indices = np.argsort(probs)[::-1]
        top_emotions = []
        for idx in sorted_indices[:top_k]:
            if probs[idx] > 0.01:  # skip near-zero
                top_emotions.append({
                    "emotion": EMOTION_LABELS[idx],
                    "confidence": round(float(probs[idx]), 4),
                })

        # Attention weights mapped to words
        display_tokens = tokens[:length]
        attn_display = attn[:len(display_tokens)]
        word_attention = [
            {"word": w, "weight": round(float(a), 4)}
            for w, a in zip(display_tokens, attn_display)
        ]

        return {
            "text": text,
            "top_emotions": top_emotions,
            "all_emotion_scores": all_scores,
            "intensity": round(intensity_val, 4),
            "attention_weights": word_attention,
            "raw_probs": probs,
        }

    def predict_batch(self, texts: List[str], top_k: int = 5) -> List[Dict]:
        """Run inference on a list of texts."""
        return [self.predict(t, top_k) for t in texts]
