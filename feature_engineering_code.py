# Feature Engineering Code
# Copy and paste this into your pipeline

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def engineer_features(df):
    """Apply feature engineering transformations"""
    df = df.copy()

    # BINNING
    # Delivery_Time_min - Quantile binning
    df['Delivery_Time_min_binned'] = pd.qcut(df['Delivery_Time_min'], q=5, labels=False, duplicates='drop')

    return df

# Apply to your data:
# df_engineered = engineer_features(df)