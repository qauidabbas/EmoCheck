"""
train_stage2.py
Driver script to generate synthetic data and fine-tune T5-small for emotion explanation.
Run from project root: python train_stage2.py
"""
import os
import sys
import json
import warnings
import torch

warnings.filterwarnings("ignore")

# Make sure src/ is importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Stage2Config
from src.stage2_dataset import SyntheticEmotionDatasetGenerator
from src.stage2_model import Stage2T5Trainer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Stage 2 T5 Explainer")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=None, help="Number of synthetic samples")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()

    cfg = Stage2Config()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.samples is not None:
        cfg.num_synthetic_samples = args.samples
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Stage 2 on: {device}")

    # 1. Generate or load synthetic dataset
    os.makedirs(os.path.dirname(cfg.dataset_path), exist_ok=True)

    if os.path.exists(cfg.dataset_path):
        print(f"📥 Loading existing synthetic dataset from {cfg.dataset_path}...")
        data = SyntheticEmotionDatasetGenerator.load(cfg.dataset_path)
        print(f"  Loaded {len(data)} samples.")
    else:
        print("🔨 Generating synthetic dataset...")
        generator = SyntheticEmotionDatasetGenerator(num_samples=cfg.num_synthetic_samples)
        data = generator.generate()
        generator.save(cfg.dataset_path)

    # 2. Split into train/val (90/10)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"📦 Train: {len(train_data)} | Val: {len(val_data)}")

    # 3. Train
    trainer = Stage2T5Trainer(cfg, device)
    trainer.train(train_data, val_data)

    print("\n✅ Stage 2 training complete. Model saved to", cfg.model_path)


if __name__ == "__main__":
    main()
