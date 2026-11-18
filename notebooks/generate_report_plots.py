"""
generate_report_plots.py (OFFLINE VERSION)
Generates metrics & plots using a curated list of test sentences.
No dataset download required.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, mean_absolute_error
from torch.utils.data import DataLoader

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Stage1Config, EMOTION_LABELS
from src.stage1_model import EmotionDetector
from src.preprocessing import Vocabulary
from src.stage1_dataset import EmotionDataset, collate_emotion

# ─────────────────────────────────────────────────────────────
# 1. LOAD MODEL & VOCAB
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = Stage1Config()
print(f"🚀 Loading model from {CFG.vocab_path}...")
VOCAB = Vocabulary.load(CFG.vocab_path)

model = EmotionDetector(
    vocab_size=len(VOCAB), embed_dim=CFG.embed_dim,
    hidden_dim=CFG.hidden_dim, num_classes=CFG.num_classes,
    num_layers=CFG.num_lstm_layers, dropout=CFG.dropout, pad_idx=0
)
ckpt = torch.load(CFG.model_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
print("✅ Model loaded successfully.")

# ─────────────────────────────────────────────────────────────
# 2. HARDCODED OFFLINE TEST DATA (50 Sentences)
#    This replaces the dataset download.
#    Indices: 0:admiration ... 17:joy ... 25:sadness ... 27:neutral
# ─────────────────────────────────────────────────────────────
print("📝 Using offline test data (50 sentences)...")
TEST_DATA = [
    # Positive
    ("I am so happy today!", [17]), ("This is the best day ever.", [17, 0]),
    ("I feel absolutely amazing.", [13, 17]), ("I love this so much.", [18]),
    ("Great job!", [4]), ("I am grateful for your help.", [15]),
    ("I'm proud of what we did.", [21]), ("I'm excited to go!", [13]),
    ("I feel optimistic about it.", [20]), ("I realized something important.", [22]),
    ("This made me smile.", [1]), ("I want this badly.", [8]),
    
    # Negative
    ("I am furious about this.", [2]), ("I hate this mess.", [11]),
    ("I am very disappointed.", [9]), ("This is unacceptable.", [10]),
    ("I feel so sad and lonely.", [25, 16]), ("I miss her so much.", [16]),
    ("I'm scared of what's next.", [14]), ("I feel really nervous.", [19]),
    ("This is ridiculous.", [3]), ("I am annoyed.", [3]),
    ("I feel sick.", [11]), ("I'm embarrassed by my mistake.", [12]),
    ("I'm confused about this.", [6]), ("I'm angry and sad.", [2, 25]),
    
    # Mixed / Complex
    ("I love him but he is toxic.", [18, 3]),
    ("I won but my friend lost.", [17, 25]),
    ("It's funny but sad.", [1, 25]),
    ("I'm happy but nervous.", [17, 19]),
    ("I'm relieved it's over.", [23]),
    ("I feel curious but wary.", [7, 14]),
    ("I'm grateful but sad.", [15, 25]),
    
    # Neutral / Low Energy
    ("I feel nothing.", [27]),
    ("It's just a normal day.", [27]),
    ("I don't have strong feelings.", [27]),
    ("The sky is blue.", [27]),
    ("I am sitting here.", [27]),
    ("This is just text.", [27]),
    ("I am reading a book.", [7]),
    ("I'm bored.", [27]),
    ("I'm waiting.", [27]),
    ("I'm listening to music.", [1]),
    
    # Hard / Sarcasm
    ("Oh great, another problem.", [3, 10]),
    ("Just what I needed.", [3]),
    ("I'm sure this will work fine.", [27]),
    ("I didn't expect that.", [26]),
    ("Wow, really?", [26]),
    ("I'm shocked.", [26]),
    ("I don't believe this.", [6]),
    ("This is weird.", [6]),
    ("I'm curious.", [7]),
]

test_texts = [x[0] for x in TEST_DATA]
test_labels = [x[1] for x in TEST_DATA]

test_dataset = EmotionDataset(test_texts, test_labels, VOCAB, max_len=CFG.max_seq_len)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_emotion)

# ─────────────────────────────────────────────────────────────
# 3. RUN INFERENCE
# ─────────────────────────────────────────────────────────────
print("🧠 Running inference...")
all_probs, all_preds, all_labels, all_int_true, all_int_pred = [], [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        ids = batch["ids"].to(DEVICE)
        lengths = batch["lengths"].to(DEVICE)
        labels = batch["labels"].cpu().numpy()
        intensities = batch["intensities"].cpu().numpy()

        logits, pred_int, _ = model(ids, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels)
        all_int_true.extend(intensities)
        all_int_pred.extend(pred_int.cpu().numpy())

all_probs = np.concatenate(all_probs)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_int_true = np.array(all_int_true)
all_int_pred = np.array(all_int_pred)

# ─────────────────────────────────────────────────────────────
# 4. COMPUTE METRICS
# ─────────────────────────────────────────────────────────────
f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
int_mae = mean_absolute_error(all_int_true, all_int_pred)

print("\n" + "="*60)
print("📊 STAGE 1 TEST METRICS (Offline)")
print("="*60)
print(f"Macro F1:       {f1_macro:.4f}")
print(f"Weighted F1:    {f1_weighted:.4f}")
print(f"Intensity MAE:  {int_mae:.4f}")
print(f"Test Samples:   {len(test_texts)} (Curated Offline)")
print("="*60)

# ─────────────────────────────────────────────────────────────
# 5. GENERATE PLOTS
# ─────────────────────────────────────────────────────────────
os.makedirs("notebooks/figures", exist_ok=True)

# 5.1 Loss Curve (Skip if no history, or use dummy data for visual)
hist_path = "notebooks/training_history.json"
if os.path.exists(hist_path):
    with open(hist_path) as f:
        hist = json.load(f)
    plt.figure(figsize=(7,4))
    plt.plot(hist["epoch"], hist["train_loss"], "b-o", label="Train Loss")
    plt.plot(hist["epoch"], hist["val_loss"], "r-s", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("notebooks/figures/loss_curve.png", dpi=300)
    plt.close()
else:
    print("⚠️ No training_history.json found. Skipping Loss Curve.")

# 5.2 PR Curves (Top 5 emotions)
top5 = np.argsort(np.sum(all_labels, axis=0))[::-1][:5]
plt.figure(figsize=(8,5))
for idx in top5:
    prec, rec, _ = precision_recall_curve(all_labels[:, idx], all_probs[:, idx])
    ap = average_precision_score(all_labels[:, idx], all_probs[:, idx])
    plt.plot(rec, prec, label=f"{EMOTION_LABELS[idx]} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("notebooks/figures/pr_curves.png", dpi=300)
plt.close()

# 5.3 Intensity Scatter
plt.figure(figsize=(6,6))
plt.scatter(all_int_true, all_int_pred, alpha=0.6, edgecolor="k", s=50)
plt.plot([0,1], [0,1], "r--", lw=2)
plt.xlabel("True Intensity (Proxy)"); plt.ylabel("Predicted Intensity")
plt.title(f"Intensity Regression (MAE={int_mae:.3f})")
plt.xlim(0,1); plt.ylim(0,1); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/figures/intensity_plot.png", dpi=300)
plt.close()

print("\n✅ All figures saved to: notebooks/figures/")
print("📂 Open that folder to view/download for your report.")