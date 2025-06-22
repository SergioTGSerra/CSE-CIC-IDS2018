#!/usr/bin/env python3
"""
Test script for the machine learning pipeline.
This script loads a sample dataset and runs a simplified version of the pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_sample_data(sample_size=200):
    """
    Create a sample dataset for testing purposes.
    """
    # Create sample features
    np.random.seed(42)
    
    sample_data = {
        'Flow Duration': np.random.randint(1, 100000, sample_size),
        'Total Fwd Packets': np.random.randint(1, 100, sample_size),
        'Total Backward Packets': np.random.randint(1, 100, sample_size),
        'Total Length of Fwd Packets': np.random.randint(1, 10000, sample_size),
        'Total Length of Bwd Packets': np.random.randint(1, 10000, sample_size),
        'Fwd Packet Length Max': np.random.randint(1, 1500, sample_size),
        'Fwd Packet Length Min': np.random.randint(0, 100, sample_size),
        'Fwd Packet Length Mean': np.random.uniform(10, 500, sample_size),
        'Fwd Header Length.1': np.random.randint(20, 100, sample_size),  # Duplicate column
        'Fwd Header Length': np.random.randint(20, 100, sample_size),
        'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], sample_size),
        'Destination Port': np.random.choice([80, 443, 22, 53, 8080], sample_size),
        'Label': np.random.choice(['BENIGN', 'DoS', 'PortScan', 'Brute Force', 'Web Attack'], sample_size, 
                                 p=[0.7, 0.1, 0.1, 0.05, 0.05])
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    print(f"Created sample dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

def normalize_column_names(df):
    """
    Normalize column names by removing leading/trailing whitespace,
    replacing spaces with underscores, and converting to lowercase.
    """
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    print("Column name mapping:")
    for orig, norm in zip(original_columns, df.columns):
        if orig != norm:
            print(f"  {orig} -> {norm}")
    
    return df

def remove_duplicate_columns(df):
    """
    Identify and remove duplicate columns from the DataFrame.
    """
    columns = df.columns.tolist()
    duplicate_columns = []
    
    for col in columns:
        if col.endswith(('.1', '.2', '.3', '.4', '.5')):
            base_col = col.rsplit('.', 1)[0]
            if base_col in columns:
                if df[col].equals(df[base_col]):
                    duplicate_columns.append(col)
                    print(f"Found duplicate column: {col} (duplicate of {base_col})")
    
    if duplicate_columns:
        df = df.drop(columns=duplicate_columns)
        print(f"Removed {len(duplicate_columns)} duplicate columns")
    else:
        print("No duplicate columns found")
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using Label Encoding.
    """
    df_encoded = df.copy()
    label_encoders = {}
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'label' in categorical_columns:
        categorical_columns.remove('label')
    
    print(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    if 'label' in df.columns:
        le = LabelEncoder()
        df_encoded['label'] = le.fit_transform(df['label'].astype(str))
        label_encoders['label'] = le
        print(f"Encoded label: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df_encoded, label_encoders

def main():
    """Main function to test the pipeline."""
    print("Testing Machine Learning Pipeline")
    print("-" * 40)
    
    # Load sample data
    df = load_sample_data(200)
    print(f"Sample data shape: {df.shape}")
    
    # Normalize column names
    df = normalize_column_names(df)
    
    # Remove duplicate columns
    df = remove_duplicate_columns(df)
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df)
    
    # Convert all data types to float (except label)
    columns = df_encoded.columns.tolist()
    if 'label' in columns:
        columns.remove('label')
    
    for col in columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').astype(float)
    
    print("Data types after conversion:")
    print(df_encoded.dtypes.head())
    
    # Split dataset into features and target
    X = df_encoded.drop(columns=['label'])
    y = df_encoded['label']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nPipeline test completed successfully!")

if __name__ == "__main__":
    main()