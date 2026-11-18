"""
Stage 1 Trainer: trains EmotionDetector with multi-label BCE + intensity MSE.
Includes gradient clipping, LR scheduling, early stopping, checkpointing.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, classification_report

from .config import Stage1Config, EMOTION_LABELS
from .stage1_model import EmotionDetector


class Stage1Trainer:
    def __init__(self, model: EmotionDetector, cfg: Stage1Config, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.int_criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        # 🆕 ADDED: Initialize training history dictionary
        self.history = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "train_f1_macro": [], "val_f1_macro": [], "val_int_mae": []
        }

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        self.model.train() if train else self.model.eval()

        total_cls_loss, total_int_loss, total_samples = 0.0, 0.0, 0

        all_preds, all_labels = [], []

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in loader:
                ids = batch["ids"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                labels = batch["labels"].to(self.device).float()
                intensities = batch["intensities"].to(self.device).float()

                logits, pred_intensity, _ = self.model(ids, lengths)

                cls_loss = self.cls_criterion(logits, labels)
                int_loss = self.int_criterion(pred_intensity, intensities)

                loss = self.cfg.cls_loss_weight * cls_loss + self.cfg.int_loss_weight * int_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

                bs = ids.size(0)
                total_cls_loss += cls_loss.item() * bs
                total_int_loss += int_loss.item() * bs
                total_samples += bs

                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        avg_cls = total_cls_loss / total_samples
        avg_int = total_int_loss / total_samples
        avg_total = self.cfg.cls_loss_weight * avg_cls + self.cfg.int_loss_weight * avg_int

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        return {
            "total_loss": avg_total,
            "cls_loss": avg_cls,
            "int_loss": avg_int,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print("\n" + "=" * 60)
        print("  STAGE 1 TRAINING")
        print("=" * 60)

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_m = self._run_epoch(train_loader, True)
            val_m = self._run_epoch(val_loader, False)

            self.scheduler.step(val_m["total_loss"])
            
            # 🆕 ADDED: Log metrics for plotting
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_m["total_loss"])
            self.history["val_loss"].append(val_m["total_loss"])
            self.history["train_f1_macro"].append(train_m["f1_macro"])
            self.history["val_f1_macro"].append(val_m["f1_macro"])
            self.history["val_int_mae"].append(0.0)  # Placeholder (MAE computed in evaluate())

            print(f"\nEpoch {epoch}/{self.cfg.num_epochs}")
            print(f"  Train — loss: {train_m['total_loss']:.4f} | F1m: {train_m['f1_macro']:.4f}")
            print(f"  Val   — loss: {val_m['total_loss']:.4f} | F1m: {val_m['f1_macro']:.4f}")

            if val_m["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_m["total_loss"]
                self.patience_counter = 0
                self._save_checkpoint()
                print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.patience:
                    print(f"  ✗ Early stopping at epoch {epoch}")
                    break

        self._load_checkpoint()
        print("\nTraining complete. Best model loaded.")
        
        # 🆕 ADDED: Save training history to JSON for plots
        os.makedirs("notebooks", exist_ok=True)
        with open("notebooks/training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print("  ✓ Training history saved to notebooks/training_history.json")

    def _save_checkpoint(self):
        os.makedirs(os.path.dirname(self.cfg.model_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, self.cfg.model_path)

    def _load_checkpoint(self):
        ckpt = torch.load(self.cfg.model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()

        all_preds, all_labels = [], []
        all_int_pred, all_int_true = [], []

        with torch.no_grad():
            for batch in test_loader:
                ids = batch["ids"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                labels = batch["labels"].to(self.device)
                intensities = batch["intensities"].to(self.device)

                logits, pred_int, _ = self.model(ids, lengths)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                all_int_pred.extend(pred_int.cpu().numpy())
                all_int_true.extend(intensities.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print("\n" + "=" * 60)
        print("  STAGE 1 — TEST EVALUATION")
        print("=" * 60)

        print("\nClassification Report:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=EMOTION_LABELS,
            zero_division=0
        ))

        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        int_mae = np.mean(np.abs(np.array(all_int_pred) - np.array(all_int_true)))

        print(f"\nMacro F1: {f1_macro:.4f}")
        print(f"Intensity MAE: {int_mae:.4f}")

        return {
            "f1_macro": f1_macro,
            "intensity_mae": int_mae,
        }