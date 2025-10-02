import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Business Impact Analysis: Food Delivery Time Prediction System', 
             fontsize=18, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(gs[0, 0])
categories = ['Before ML', 'After ML']
nps_scores = [42, 56]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(categories, nps_scores, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('NPS Score', fontsize=11, fontweight='bold')
ax1.set_title('Customer Satisfaction\n+33% Improvement', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 70])
for bar, score in zip(bars, nps_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{score}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
labels = ['Before', 'After']
sizes = [18, 8]
explode = (0.1, 0)
colors_pie = ['#FF6B6B', '#51CF66']
ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Late Deliveries\n-56% Reduction', fontsize=12, fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
utilization = [72, 85]
bars = ax3.bar(categories, utilization, color=['#FFD93D', '#6BCF7F'], 
               edgecolor='black', linewidth=2, width=0.6)
ax3.set_ylabel('Utilization %', fontsize=11, fontweight='bold')
ax3.set_title('Driver Utilization\n+18% Increase', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 100])
for bar, util in zip(bars, utilization):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{util}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax4 = fig.add_subplot(gs[1, :2])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
before_tickets = [1200, 1180, 1220, 1190, 1210, 1200]
after_tickets = [1200, 1100, 950, 820, 750, 720]
x = np.arange(len(months))
ax4.plot(x, before_tickets, 'ro-', linewidth=2, markersize=8, label='Before ML')
ax4.plot(x, after_tickets, 'go-', linewidth=2, markersize=8, label='After ML')
ax4.set_xticks(x)
ax4.set_xticklabels(months)
ax4.set_ylabel('Daily Support Tickets', fontsize=12, fontweight='bold')
ax4.set_title('Support Ticket Reduction (-40% after 6 months)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
categories_roi = ['Cost\nSavings', 'Revenue\nIncrease', 'Total\nImpact']
values = [1.2, 1.2, 2.4]
colors_roi = ['#51CF66', '#4ECDC4', '#FFD93D']
bars = ax5.bar(categories_roi, values, color=colors_roi, edgecolor='black', linewidth=2)
ax5.set_ylabel('Annual Value ($M)', fontsize=11, fontweight='bold')
ax5.set_title('Estimated Annual ROI', fontsize=12, fontweight='bold')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'${val:.1f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax6 = fig.add_subplot(gs[2, :])
distance_ranges = ['0-2 km', '2-5 km', '5-10 km', '10+ km']
accuracy = [95.2, 92.8, 88.5, 83.1]
sample_sizes = [15234, 23451, 12098, 4217]
x_pos = np.arange(len(distance_ranges))

bars = ax6.bar(x_pos, accuracy, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(distance_ranges))),
               edgecolor='black', linewidth=2, width=0.6)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(distance_ranges)
ax6.set_ylabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Delivery Distance Range', fontsize=12, fontweight='bold')
ax6.set_title('Model Accuracy by Distance Segment', fontsize=13, fontweight='bold')
ax6.set_ylim([75, 100])

for i, (bar, acc, samples) in enumerate(zip(bars, accuracy, sample_sizes)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1f}%\n(n={samples:,})', ha='center', va='bottom', 
             fontsize=10, fontweight='bold')

plt.savefig('../assets/business_impact.png', dpi=300, bbox_inches='tight')
print("✅ Created: assets/business_impact.png")
