import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(42)

hours = list(range(24))
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

error_matrix = np.random.uniform(2, 8, (24, 7))
error_matrix[11:14, :] += np.random.uniform(2, 4, (3, 7))
error_matrix[17:21, :] += np.random.uniform(3, 6, (4, 7))
error_matrix[6:11, 5:7] -= 1.5

error_df = pd.DataFrame(error_matrix, 
                        index=[f'{h:02d}:00' for h in hours],
                        columns=days)

fig, ax = plt.subplots(figsize=(14, 10))

sns.heatmap(error_df, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Average Absolute Error (minutes)'},
            linewidths=0.5, linecolor='gray', ax=ax,
            vmin=2, vmax=12, center=7)

ax.set_title('Average Prediction Error by Day and Hour\n(Darker = Higher Error)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', fontsize=13, fontweight='bold')
ax.set_ylabel('Hour of Day', fontsize=13, fontweight='bold')

ax.text(3.5, 13, '← Lunch Rush', fontsize=10, ha='center', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
ax.text(3.5, 19, '← Dinner Rush', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

plt.tight_layout()
plt.savefig('../assets/error_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Created: assets/error_heatmap.png")
