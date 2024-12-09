import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("preprocessed_data.csv")
data.info()

# Feature engineering
data["date"] = pd.to_datetime(data["date"], errors="coerce")

# Sort the data by store_nbr, item_nbr, and date
data = data.sort_values(by=["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# Define parameters
N = 7  # Number of past days to consider
target = "unit_sales"

# Step 1: Feature Engineering
# Temporal Features
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["week_of_year"] = data["date"].dt.isocalendar().week
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

# Create lag features for past N days
for lag in range(1, N + 1):
    # Unit Sales Lag: Fill NaN with 0
    data[f"unit_sales_lag_{lag}"] = (
        data.groupby(["store_nbr", "item_nbr"])[target]
        .shift(lag)
        .fillna(0)  # Fill NaN with 0
    )

    # Oil Price Lag: Fill NaN with the previous valid value (forward fill within each store group)
    data[f"oil_price_lag_{lag}"] = (
        data.groupby(["store_nbr"])["oil_price"].shift(lag).ffill().bfill()
    )

# Rolling Statistics: Ensure correct index alignment
rolling_sales = (
    data.groupby(["store_nbr", "item_nbr"])[target]
    .rolling(N)
    .agg(["mean", "std"])
    .reset_index(level=["store_nbr", "item_nbr"], drop=True)
)

# Add rolling statistics back to the original DataFrame
data["rolling_sales_mean"] = rolling_sales["mean"].fillna(0)
data["rolling_sales_std"] = rolling_sales["std"].fillna(0)


string_columns = data.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in string_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

print("\nString attributes converted to class features:")


train_end_date = "2017-06-01"
test_start_date = "2017-06-01"

# Split the dataset into train and test based on the date
train_df = data[data["date"] <= train_end_date]
filtered_test_df = data[data["date"] >= test_start_date]

# Find unique item-store pairs in the train set
train_item_store_pairs = set(zip(train_df["item_nbr"], train_df["store_nbr"]))

# Filter the test set to include only item-store pairs present in the train set
test_df = filtered_test_df[
    filtered_test_df.apply(
        lambda x: (x["item_nbr"], x["store_nbr"]) in train_item_store_pairs, axis=1
    )
]

# Identify test rows excluded by the filter
missing_test_rows = filtered_test_df[
    ~filtered_test_df.apply(
        lambda x: (x["item_nbr"], x["store_nbr"]) in train_item_store_pairs, axis=1
    )
]

# Add the missing test rows back to the train set
train_df = pd.concat([train_df, missing_test_rows], ignore_index=True)


import pandas as pd
from autogluon.tabular import TabularPredictor

target = "unit_sales"

train_df[target] = np.log1p(train_df[target])
test_df[target] = np.log1p(test_df[target])
train_df = train_df.drop(columns=["date", "id"])  # Drop non-predictive columns
test_df = test_df.drop(columns=["date", "id"])
print(test_df)

# Initialize AutoGluon Predictor
# predictor = TabularPredictor.load("/home/ec2-user/SageMaker/SDR_VLM_Version2/maunilv_sandbox/autogluon/AutogluonModels/ag-20241207_082316")
predictor = TabularPredictor.load("/home/ec2-user/SageMaker/SDR_VLM_Version2/maunilv_sandbox/autogluon/AutogluonModels/ag-20241207_085941")
# Evaluate on the test set
performance = predictor.evaluate(test_df)
print("Model Performance on Test Set:")
print(performance)

# Predict on the test data
predictions_log = predictor.predict(test_df)

# Reverse the log transformation to get predictions in the original scale
predictions = np.expm1(predictions_log)

print("\nSample Predictions (Original Scale):")
print(predictions.head())

# Reverse log transformation for the actual test target for evaluation
actuals = np.expm1(test_df[target])


mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
print(f"\nModel Performance on Original Scale:\nMSE: {mse:.4f}, MAE: {mae:.4f}")

# # Feature importance analysis
# feature_importance = predictor.feature_importance(test_df)
# print("\nFeature Importance:")
# print(feature_importance)


# Load the trained predictor
predictor = TabularPredictor.load(
    "/home/ec2-user/SageMaker/SDR_VLM_Version2/maunilv_sandbox/autogluon/AutogluonModels/ag-20241207_085941"
)

# Get the leaderboard of all models
leaderboard_df = predictor.leaderboard(test_df, silent=True)
print("\nLeaderboard of all models:")
print(leaderboard_df)

# Evaluate each model individually on the test set
results = []
for model_name in leaderboard_df["model"]:
    print(f"\nEvaluating model: {model_name}")
    performance = predictor.evaluate(test_df, model=model_name)
    print(performance)

    # Get predictions and reverse log transformation
    predictions_log = predictor.predict(test_df, model=model_name)
    predictions = np.expm1(predictions_log)

    # Reverse log transformation for actuals
    actuals = np.expm1(test_df[target])

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    # Append results
    results.append({"Model": model_name, "Test MSE": mse, "Test MAE": mae})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Print the performance of all models
print("\nTest Performance of All Models:")
print(results_df)

# Save to a CSV file for future reference
results_df.to_csv("autogluon_model_performance.csv", index=False)
