import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Define paths
PROCESSED_DATA_PATH = "../data/processed/"
FEATURES_DATA_PATH = "../data/features/"

# Ensure the feature directory exists
os.makedirs(FEATURES_DATA_PATH, exist_ok=True)

def feature_engineering(df, filename):
    print(f" Processing {filename} for feature engineering...")

    # Handling missing values before transformations
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical NaNs with mode
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Fill numerical NaNs with median

    # Handling categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Creating new time-based features (if timestamp column exists)
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['weekday'] = df['timestamp'].dt.weekday
        df.drop(columns=['timestamp'], inplace=True)

    # Generating statistical features (rolling averages, min/max values)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[f"{col}_mean"] = df[col].rolling(window=5, min_periods=1).mean()
        df[f"{col}_std"] = df[col].rolling(window=5, min_periods=1).std()
        df[f"{col}_min"] = df[col].rolling(window=5, min_periods=1).min()
        df[f"{col}_max"] = df[col].rolling(window=5, min_periods=1).max()

    # Standardize numerical features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f" Feature Engineering Completed for {filename}")
    return df

# Process all preprocessed datasets
for file in os.listdir(PROCESSED_DATA_PATH):
    if file.endswith('.csv'):
        file_path = os.path.join(PROCESSED_DATA_PATH, file)
        df = pd.read_csv(file_path)

        # Apply feature engineering
        df_transformed = feature_engineering(df, file)

        # Save transformed dataset
        output_file = os.path.join(FEATURES_DATA_PATH, f"features_{file}")
        df_transformed.to_csv(output_file, index=False)
        print(f" Saved: {output_file}")

print(" Feature Engineering Completed for All Files.")
