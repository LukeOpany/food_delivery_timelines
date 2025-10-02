import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

epochs = np.arange(1, 51)
train_loss = 12 * np.exp(-0.08 * epochs) + np.random.normal(0, 0.3, 50)
val_loss = 13 * np.exp(-0.07 * epochs) + np.random.normal(0, 0.4, 50) + 0.5

train_r2 = 1 - train_loss / train_loss[0]
val_r2 = 1 - val_loss / val_loss[0]

lr = 0.1 * np.exp(-0.05 * epochs)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (RMSE)', fontsize=12, fontweight='bold')
ax1.set_title('Model Training Progress: Loss Curves', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, train_r2, 'b-', linewidth=2, label='Training R²', marker='o', markersize=3)
ax2.plot(epochs, val_r2, 'r-', linewidth=2, label='Validation R²', marker='s', markersize=3)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Model Training Progress: R² Score', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

ax3.plot(epochs, lr, 'g-', linewidth=2.5, marker='d', markersize=4)
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax3.set_title('Learning Rate Schedule (Exponential Decay)', fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

gap = val_loss - train_loss
ax4.fill_between(epochs, 0, gap, alpha=0.3, color='red', label='Generalization Gap')
ax4.plot(epochs, gap, 'r-', linewidth=2, marker='o', markersize=3)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('Validation Loss - Training Loss', fontsize=12, fontweight='bold')
ax4.set_title('Overfitting Detection (Lower Gap = Better)', fontsize=14, fontweight='bold', pad=15)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/training_progress.png', dpi=300, bbox_inches='tight')
print("✅ Created: assets/training_progress.png")
