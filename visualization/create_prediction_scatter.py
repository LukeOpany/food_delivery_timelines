import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)
n_samples = 1000

y_true = np.random.gamma(shape=3, scale=8, size=n_samples) + 10
y_true = np.clip(y_true, 10, 70)

noise = np.random.normal(0, 3, n_samples)
y_pred = y_true + noise

mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

scatter = ax1.scatter(y_true, y_pred, alpha=0.4, s=30, c=np.abs(y_true-y_pred), 
                      cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
         'r--', lw=3, label='Perfect Prediction', alpha=0.8)

margin = 5
ax1.fill_between([y_true.min(), y_true.max()], 
                 [y_true.min()-margin, y_true.max()-margin],
                 [y_true.min()+margin, y_true.max()+margin],
                 alpha=0.2, color='green', label=f'±{margin} min tolerance')

ax1.set_xlabel('Actual Delivery Time (minutes)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Predicted Delivery Time (minutes)', fontsize=13, fontweight='bold')
ax1.set_title(f'Prediction Accuracy: Actual vs Predicted\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}', 
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Absolute Error (minutes)', fontsize=11, fontweight='bold')

residuals = y_true - y_pred
ax2.scatter(y_pred, residuals, alpha=0.4, s=30, c=np.abs(residuals), 
            cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='red', linestyle='--', lw=2, label='Zero Error')
ax2.axhline(y=margin, color='orange', linestyle=':', lw=2, alpha=0.7)
ax2.axhline(y=-margin, color='orange', linestyle=':', lw=2, alpha=0.7, label=f'±{margin} min')

ax2.set_xlabel('Predicted Delivery Time (minutes)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=13, fontweight='bold')
ax2.set_title('Residual Analysis\n(Random pattern indicates good fit)', 
              fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/prediction_accuracy.png', dpi=300, bbox_inches='tight')
print("✅ Created: assets/prediction_accuracy.png")
