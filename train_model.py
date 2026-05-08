"""Train and evaluate a food delivery time regression model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).with_name("Food_Delivery_Times.csv")
TARGET_COLUMN = "Delivery_Time_min"
DROP_COLUMNS = ["Order_ID"]
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the delivery time dataset."""
    return pd.read_csv(path)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build the preprocessing and model pipeline."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> dict[str, float]:
    """Train the model and return test-set metrics."""
    X = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, predictions),
    }


def main() -> None:
    df = load_data()
    metrics = train_and_evaluate(df)

    print("Food Delivery Time Model")
    print("========================")
    print(f"Rows: {len(df):,}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"MAE:  {metrics['mae']:.2f} minutes")
    print(f"RMSE: {metrics['rmse']:.2f} minutes")
    print(f"R2:   {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
