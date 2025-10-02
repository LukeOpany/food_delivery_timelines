import os

os.makedirs('../assets', exist_ok=True)

print("=" * 60)
print("📊 Generating Visual Assets")
print("=" * 60)
print()

scripts = [
    'create_model_comparison.py',
    'create_feature_importance.py',
    'create_prediction_scatter.py',
    'create_error_heatmap.py',
    'create_training_progress.py',
    'create_business_impact.py',
    'create_interactive_dashboard.py'
]

for i, script in enumerate(scripts, 1):
    print(f"[{i}/{len(scripts)}] Running {script}...")
    try:
        exec(open(script).read())
        print(f"    ✅ Success")
    except Exception as e:
        print(f"    ❌ Error: {e}")
    print()

print("=" * 60)
print("🎉 Complete!")
print("=" * 60)
