import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define paths
RAW_DATA_PATHS = {
    "assuremoss": "../data/assuremoss_dataset/",
    "horizontal_scaling": "../data/horizontal_scaling_dataset/"
}
PROCESSED_DATA_PATH = "../data/processed/"

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def preprocess_data(file_path):
    print(f" Loading: {file_path}")
    df = pd.read_csv(file_path)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)  # Categorical: Fill with mode
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Numerical: Fill with median

    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print(f" Processed {file_path}")
    return df

# Process datasets dynamically
for category, folder_path in RAW_DATA_PATHS.items():
    if not os.path.exists(folder_path):
        print(f" Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df_processed = preprocess_data(file_path)
            
            # Save processed file
            output_file = os.path.join(PROCESSED_DATA_PATH, f"processed_{file}")
            df_processed.to_csv(output_file, index=False)
            print(f" Saved: {output_file}")

print(" Data Preprocessing Completed.")
