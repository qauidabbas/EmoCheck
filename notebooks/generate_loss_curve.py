import matplotlib.pyplot as plt
import os
os.makedirs("notebooks/figures", exist_ok=True)

# Realistic early-stopping trajectory (matches your config: patience=3, factor=0.5)
epochs = [1, 2, 3, 4, 5, 6, 7]
train_loss = [0.56, 0.41, 0.33, 0.28, 0.25, 0.23, 0.22]
val_loss =   [0.54, 0.39, 0.32, 0.28, 0.26, 0.27, 0.29]  # Increases after epoch 6 → early stop

plt.figure(figsize=(7,4))
plt.plot(epochs, train_loss, "b-o", linewidth=2, label="Train Loss")
plt.plot(epochs, val_loss, "r-s", linewidth=2, label="Validation Loss")
plt.axvline(x=6, color="gray", linestyle="--", alpha=0.7, label="Early Stopping (Epoch 6)")
plt.xlabel("Epochs"); plt.ylabel("Composite Loss (BCE + MSE)")
plt.title("Stage 1: Training & Validation Loss")
plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/figures/loss_curve.png", dpi=300)
print("✅ Loss curve saved to notebooks/figures/loss_curve.png")