"""
train_stage1.py
Driver script to load GoEmotions, build vocab, create dataloaders, and train Stage 1.
Run from project root: python train_stage1.py
"""
import os
import sys
import warnings
import torch
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

# Make sure src/ is importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Stage1Config, EMOTION_LABELS
from src.preprocessing import Vocabulary
from src.stage1_dataset import EmotionDataset, collate_emotion
from src.stage1_model import EmotionDetector
from src.stage1_train import Stage1Trainer

def load_goemotions():
    print("📥 Loading GoEmotions dataset from HuggingFace...")
    from datasets import load_dataset
    dataset = load_dataset("dair-ai/go_emotions", "full")
    
    texts, labels = [], []
    for split in ["train", "validation"]:
        for item in dataset[split]:
            if len(item["labels"]) > 0:  # Skip empty annotations
                texts.append(item["text"])
                labels.append(item["labels"])
    return texts, labels

def prepare_data(texts, labels, vocab, cfg):
    print("📊 Splitting data (90% train / 10% val)...")
    split_idx = int(0.9 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = EmotionDataset(train_texts, train_labels, vocab, cfg.max_seq_len)
    val_dataset = EmotionDataset(val_texts, val_labels, vocab, cfg.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_emotion)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_emotion)
    
    return train_loader, val_loader

def main():
    cfg = Stage1Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 1. Load data
    texts, labels = load_goemotions()
    print(f"✅ Loaded {len(texts)} samples.")

    # 2. Build vocabulary
    print("🔨 Building vocabulary...")
    vocab = Vocabulary(freq_threshold=cfg.min_word_freq)
    vocab.build(texts)
    os.makedirs(os.path.dirname(cfg.vocab_path), exist_ok=True)
    vocab.save(cfg.vocab_path)

    # 3. Prepare dataloaders
    train_loader, val_loader = prepare_data(texts, labels, vocab, cfg)
    print(f"📦 Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # 4. Initialize model
    model = EmotionDetector(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        num_layers=cfg.num_lstm_layers,
        dropout=cfg.dropout,
        pad_idx=0
    )
    print(f"🤖 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Train
    trainer = Stage1Trainer(model, cfg, device)
    trainer.train(train_loader, val_loader)

    print("\n✅ Stage 1 training complete. Checkpoints & history saved.")

if __name__ == "__main__":
    main()