import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# Step 1: Load Processed Feature Data
features_dir = "../data/features"
feature_files = [f for f in os.listdir(features_dir) if f.endswith(".csv")]

dfs = []
total_records = 0

for file in feature_files:
    file_path = os.path.join(features_dir, file)
    df = pd.read_csv(file_path)
    dfs.append(df)
    total_records += len(df)

print(f"\n Loaded {len(feature_files)} feature files with {total_records} total records.")

# Combine datasets
df = pd.concat(dfs, ignore_index=True)

# Step 2: Use Full Dataset
df = df.sample(frac=1.0, random_state=42)

# Step 3: Identify the Target Column
target_col = "label" if "label" in df.columns else None
if not target_col:
    raise ValueError("Target column 'label' not found!")

df = df.dropna(subset=[target_col])

# Step 4: Convert Categorical Features
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("category")

# Step 5: Select Features and Target
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Data split into training and testing sets.")

# Step 7: Train LightGBM Model
print("\n Training optimized LightGBM model...")

start_time = time.time()

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

#  **Further Regularized LightGBM Parameters**
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 10,  # Smaller trees to generalize better
    "min_data_in_leaf": 150,  # Prevents small overfitted leaves
    "max_depth": 4,  # Shallow trees = better generalization
    "learning_rate": 0.01,  # Even slower learning
    "feature_fraction": 0.65,  # Each tree sees only 65% of features
    "bagging_fraction": 0.7,  # Each tree sees 70% of training data
    "bagging_freq": 20,  # More frequent bagging for randomness
    "lambda_l1": 1.0,  # Stronger L1 regularization (Lasso)
    "lambda_l2": 1.0,  # Stronger L2 regularization (Ridge)
    "verbose": -1,
    "n_jobs": -1
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_test],
    num_boost_round=400,  # Lowering rounds to prevent overfitting
    callbacks=[
        lgb.early_stopping(30),  # Earlier stopping to prevent overfitting
        lgb.log_evaluation(50)
    ]
)

end_time = time.time()
training_time = end_time - start_time
print(f"\n Training completed in {training_time:.2f} seconds.")

# Step 8: Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ✅ Fixed Regression Accuracy Calculation
valid_mask = np.abs(y_test) > 1e-6  # Exclude near-zero values
filtered_y_test = y_test[valid_mask]
filtered_y_pred = y_pred[valid_mask]

if len(filtered_y_test) > 0:
    mape = np.mean(np.abs((filtered_y_test - filtered_y_pred) / filtered_y_test)) * 100
    accuracy = 100 - mape  # Higher is better
else:
    accuracy = float("nan")  # No valid values, accuracy undefined

# Retrieve RMSE values from LightGBM's best iteration
train_rmse = model.best_score["training"]["rmse"] if "training" in model.best_score else None
test_rmse = model.best_score["valid_0"]["rmse"] if "valid_0" in model.best_score else None

print("\n Model Evaluation:")
print(f" Mean Squared Error (MSE): {mse:.6f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f" R² Score: {r2:.6f}")
print(f" Regression Accuracy: {accuracy:.2f}%")  
if train_rmse is not None:
    print(f" Training RMSE: {train_rmse:.6f}")
if test_rmse is not None:
    print(f" Testing RMSE: {test_rmse:.6f}")

# Step 9: Save Model
model_path = "../models/k8s_failure_model_lgb.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print(f"\n Model saved at: {model_path}")

# Step 10: Log Training Details
log_path = "../logs/train_log_lgb.txt"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

with open(log_path, "w") as log_file:
    log_file.write(f"Training Time: {training_time:.2f} seconds\n")
    log_file.write(f"Mean Squared Error: {mse:.6f}\n")
    log_file.write(f"Root Mean Squared Error: {rmse:.6f}\n")
    log_file.write(f"R² Score: {r2:.6f}\n")
    log_file.write(f"Regression Accuracy: {accuracy:.2f}%\n") 
    if train_rmse is not None:
        log_file.write(f"Training RMSE: {train_rmse:.6f}\n")
    if test_rmse is not None:
        log_file.write(f"Testing RMSE: {test_rmse:.6f}\n")

print(f"\n Training log saved at: {log_path}")
