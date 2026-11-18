
"""
Configuration for both stages of the Emotion Intelligence pipeline.
"""
from dataclasses import dataclass

# ── 28 GoEmotions Labels ──
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

NUM_EMOTIONS = len(EMOTION_LABELS)
LABEL_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
IDX_TO_LABEL = {i: e for i, e in enumerate(EMOTION_LABELS)}

# Heuristic base intensity per emotion (used as proxy since GoEmotions has no intensity labels)
EMOTION_BASE_INTENSITY = {
    "admiration": 0.45, "amusement": 0.40, "anger": 0.80, "annoyance": 0.45,
    "approval": 0.35,   "caring": 0.50,    "confusion": 0.40, "curiosity": 0.35,
    "desire": 0.60,     "disappointment": 0.60, "disapproval": 0.45, "disgust": 0.70,
    "embarrassment": 0.55, "excitement": 0.75, "fear": 0.80, "gratitude": 0.50,
    "grief": 0.90,      "joy": 0.70,       "love": 0.80,  "nervousness": 0.55,
    "optimism": 0.45,   "pride": 0.55,     "realization": 0.35, "relief": 0.45,
    "remorse": 0.65,    "sadness": 0.65,   "surprise": 0.50, "neutral": 0.10,
}


@dataclass
class Stage1Config:
    """Hyperparameters for the emotion detection model."""
    max_seq_len: int = 128
    min_word_freq: int = 2
    embed_dim: int = 300
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    num_classes: int = 28
    dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 15
    grad_clip: float = 1.0
    patience: int = 3
    cls_loss_weight: float = 1.0
    int_loss_weight: float = 0.5
    model_path: str = "checkpoints/stage1_best.pt"
    vocab_path: str = "checkpoints/vocab.pkl"


@dataclass
class Stage2Config:
    """Hyperparameters for the emotion-to-text generator."""
    t5_model_name: str = "t5-small"
    max_input_len: int = 128
    max_target_len: int = 128
    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 10
    patience: int = 3
    num_synthetic_samples: int = 8000
    model_path: str = "checkpoints/stage2_t5/"
    dataset_path: str = "data/stage2_synthetic.json"
