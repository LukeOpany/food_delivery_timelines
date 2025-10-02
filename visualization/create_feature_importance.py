import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

features = [
    'Delivery Distance',
    'Traffic Density', 
    'Restaurant Prep Time',
    'Time of Day (Hour)',
    'Driver Experience',
    'Order Complexity',
    'Day of Week',
    'Weather Conditions',
    'Restaurant Rating',
    'Number of Items'
]
importance = [0.28, 0.19, 0.15, 0.12, 0.10, 0.08, 0.04, 0.02, 0.01, 0.01]

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
bars = ax.barh(features, importance, color=colors, edgecolor='black', linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars, importance)):
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{val:.1%}', ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('Feature Importance', fontsize=13, fontweight='bold')
ax.set_title('Top Features Driving Delivery Time Predictions\n(Based on Model Feature Importance)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim([0, max(importance) * 1.15])
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('../assets/feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Created: assets/feature_importance.png")
