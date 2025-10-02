import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

np.random.seed(42)

models = ['Baseline', 'Linear Reg', 'Random Forest', 'XGBoost', 'LightGBM']
rmse_values = [15.2, 12.8, 8.5, 7.2, 6.8]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Model Performance', 'Feature Importance', 
                    'Prediction Distribution', 'Error by Hour'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'histogram'}, {'type': 'scatter'}]]
)

fig.add_trace(go.Bar(x=models, y=rmse_values, marker_color='lightblue'), row=1, col=1)

features = ['Distance', 'Traffic', 'Prep Time', 'Hour', 'Experience']
importance = [0.28, 0.19, 0.15, 0.12, 0.10]
fig.add_trace(go.Bar(x=features, y=importance, marker_color='lightgreen'), row=1, col=2)

predictions = np.random.normal(30, 8, 1000)
fig.add_trace(go.Histogram(x=predictions, marker_color='coral', nbinsx=30), row=2, col=1)

hours = list(range(24))
errors = [3.2 + 2*np.sin(h/24 * 2*np.pi) + np.random.normal(0, 0.5) for h in hours]
fig.add_trace(go.Scatter(x=hours, y=errors, mode='lines+markers',
                         line=dict(color='purple', width=3)), row=2, col=2)

fig.update_layout(title_text="Food Delivery Timeline Prediction Dashboard",
                  title_font_size=20, showlegend=False, height=800, template='plotly_white')

fig.write_html('../assets/interactive_dashboard.html')
print("✅ Created: assets/interactive_dashboard.html")
