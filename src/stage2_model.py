
"""
Stage 2 Model: T5-small fine-tuned for emotion-to-text generation.
This is transfer learning — T5 is pretrained on C4 corpus,
and we fine-tune it on our synthetic emotion→sentence pairs.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict

from .config import Stage2Config


class EmotionToTextDataset(Dataset):
    """Dataset for T5 fine-tuning: input emotion string → output sentence."""
    def __init__(self, data: List[Dict[str, str]], tokenizer: T5Tokenizer,
                 max_input_len: int = 128, max_target_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_enc = self.tokenizer(
            item["input"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            item["target"],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_enc["input_ids"].squeeze(0)
        attention_mask = input_enc["attention_mask"].squeeze(0)
        labels = target_enc["input_ids"].squeeze(0)

        # Replace padding token id with -100 so loss ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class Stage2T5Trainer:
    """
    Fine-tunes T5-small for emotion-to-text generation.
    This is TRANSFER LEARNING: T5 was pretrained on a massive corpus,
    we adapt it to our task with our synthetic data.
    """
    def __init__(self, cfg: Stage2Config, device: torch.device):
        self.cfg = cfg
        self.device = device

        print(f"  Loading {cfg.t5_model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(cfg.t5_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            cfg.t5_model_name
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """Full training loop."""
        train_dataset = EmotionToTextDataset(
            train_data, self.tokenizer,
            self.cfg.max_input_len, self.cfg.max_target_len
        )
        val_dataset = EmotionToTextDataset(
            val_data, self.tokenizer,
            self.cfg.max_input_len, self.cfg.max_target_len
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.batch_size, shuffle=False
        )

        print("\n" + "=" * 60)
        print("  STAGE 2 TRAINING (T5 Fine-tuning — Transfer Learning)")
        print("=" * 60)
        print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        for epoch in range(1, self.cfg.num_epochs + 1):
            # --- Training ---
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_train_loss += loss.item() * input_ids.size(0)

            avg_train_loss = total_train_loss / len(train_dataset)

            # --- Validation ---
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_val_loss += outputs.loss.item() * input_ids.size(0)

            avg_val_loss = total_val_loss / len(val_dataset)
            self.scheduler.step(avg_val_loss)

            print(f"\n  Epoch {epoch}/{self.cfg.num_epochs} — "
                  f"train_loss: {avg_train_loss:.4f} | "
                  f"val_loss: {avg_val_loss:.4f}")

            # Checkpoint
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                self.save_model()
                print(f"    ✓ Saved best model (val_loss={self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.patience:
                    print(f"    ✗ Early stopping at epoch {epoch}")
                    break

            # Show sample generations every 2 epochs
            if epoch % 2 == 0:
                self._show_samples(val_data[:3])

        self.load_model()
        print("\nStage 2 training complete. Best model loaded.")

    def _show_samples(self, samples: List[Dict]):
        self.model.eval()
        print("    --- Sample Generations ---")
        for s in samples:
            generated = self.generate(s["input"])
            print(f"      IN:  {s['input']}")
            print(f"      REF: {s['target']}")
            print(f"      GEN: {generated}")
            print()

    def generate(self, input_text: str, max_length: int = 100) -> str:
        """Generate a natural sentence from an emotion input string."""
        self.model.eval()
        input_enc = self.tokenizer(
            input_text, return_tensors="pt",
            max_length=self.cfg.max_input_len, truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_enc["input_ids"],
                attention_mask=input_enc["attention_mask"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save_model(self):
        os.makedirs(self.cfg.model_path, exist_ok=True)
        self.model.save_pretrained(self.cfg.model_path)
        self.tokenizer.save_pretrained(self.cfg.model_path)

    def load_model(self):
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.cfg.model_path
        ).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.model_path)
