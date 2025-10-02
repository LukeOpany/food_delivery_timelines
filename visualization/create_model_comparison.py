import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# TODO: Replace with your actual model results from deliverytimes.ipynb
models = ['Baseline\n(Mean)', 'Linear\nRegression', 'Random\nForest', 
          'Gradient\nBoosting', 'XGBoost', 'LightGBM']
rmse = [15.2, 12.8, 8.5, 7.8, 7.2, 6.8]
mae = [12.1, 10.2, 6.8, 6.2, 5.8, 5.4]
r2 = [0.0, 0.45, 0.78, 0.82, 0.85, 0.87]
inference_time = [0.001, 0.001, 5.2, 3.8, 2.1, 1.2]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# RMSE Comparison
colors = sns.color_palette("viridis", len(models))
bars1 = ax1.bar(models, rmse, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('RMSE (minutes)', fontsize=12, fontweight='bold')
ax1.set_title('Model RMSE Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.axhline(y=min(rmse), color='red', linestyle='--', alpha=0.5, label='Best Performance')
ax1.legend()
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{rmse[i]:.1f}', ha='center', va='bottom', fontweight='bold')

# MAE Comparison
bars2 = ax2.bar(models, mae, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('MAE (minutes)', fontsize=12, fontweight='bold')
ax2.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold', pad=15)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mae[i]:.1f}', ha='center', va='bottom', fontweight='bold')

# R² Score
bars3 = ax3.bar(models, r2, color=colors, edgecolor='black', linewidth=1.2)
ax3.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax3.set_title('Model R² Score (Higher is Better)', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylim([0, 1])
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{r2[i]:.2f}', ha='center', va='bottom', fontweight='bold')

# Inference Speed
bars4 = ax4.bar(models, inference_time, color=colors, edgecolor='black', linewidth=1.2)
ax4.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
ax4.set_title('Inference Speed (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
ax4.set_yscale('log')
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height*1.2,
             f'{inference_time[i]:.1f}ms', ha='center', va='bottom', 
             fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('../assets/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Created: assets/model_comparison.png")
