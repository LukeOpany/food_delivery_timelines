# Food Delivery Time Prediction

This project predicts food delivery time in minutes from order, courier, traffic, weather, and preparation-time data.

The main goal is to show a clear machine learning workflow:

1. Load the delivery dataset.
2. Explore the features and target.
3. Fill missing values.
4. Encode categorical variables.
5. Train a regression model.
6. Evaluate prediction error in minutes.

## Dataset

The dataset is [Food_Delivery_Times.csv](Food_Delivery_Times.csv). It contains 1,000 delivery records.

| Column | Meaning |
| --- | --- |
| `Order_ID` | Unique order identifier |
| `Distance_km` | Delivery distance |
| `Weather` | Weather condition during delivery |
| `Traffic_Level` | Traffic level |
| `Time_of_Day` | Morning, afternoon, evening, or night |
| `Vehicle_Type` | Courier vehicle type |
| `Preparation_Time_min` | Restaurant preparation time |
| `Courier_Experience_yrs` | Courier experience in years |
| `Delivery_Time_min` | Target variable: actual delivery time |

## Project Files

| File | Purpose |
| --- | --- |
| [deliverytines.ipynb](deliverytines.ipynb) | Exploratory notebook with charts, preprocessing, training, and evaluation |
| [train_model.py](train_model.py) | Clean reproducible training script |
| [requirements.txt](requirements.txt) | Python dependencies |

## Notebook Workflow

The notebook is organized as a guided walkthrough:

1. Import libraries.
2. Load the dataset with a portable relative path.
3. Inspect the data with `head`, `shape`, `info`, `describe`, and missing-value checks.
4. Identify the target column and feature columns.
5. Explore the target distribution and relationships between features and delivery time.
6. Separate inputs (`X`) from the target (`y`).
7. Split the data into training and test sets.
8. Build preprocessing steps for numeric and categorical columns.
9. Train a `RandomForestRegressor`.
10. Evaluate the model with MAE, RMSE, and R2.

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3.12 -m venv --prompt food-delivery .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.12` is not available, use your installed Python 3 version:

```bash
python3 -m venv .venv
```

Run the reproducible training script:

```bash
python train_model.py
```

Or open the notebook:

```bash
jupyter notebook deliverytines.ipynb
```

When using VS Code, select the `.venv/bin/python` interpreter or notebook kernel before running cells.

## Modeling Approach

The training script uses a scikit-learn pipeline:

- Drops `Order_ID`, because it is an identifier rather than a predictive feature.
- Uses median imputation for numeric missing values.
- Uses most-frequent imputation for categorical missing values.
- One-hot encodes categorical features.
- Trains a `RandomForestRegressor`.
- Reports MAE, RMSE, and R2 on a held-out test set.

MAE is the most intuitive metric here because it answers: "On average, how many minutes off are the predictions?"

Example output:

```text
Food Delivery Time Model
========================
Rows: 1,000
Target: Delivery_Time_min
MAE:  6.72 minutes
RMSE: 9.56 minutes
R2:   0.796
```

In beginner-friendly terms, this means the model is off by about 6 to 7 minutes on average.

## Notes

The notebook is useful for learning and experimentation. The script is useful when you want a clean, repeatable result without manually running notebook cells.
