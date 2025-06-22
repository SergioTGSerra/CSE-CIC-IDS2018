#!/usr/bin/env python3
"""
Test script for the machine learning pipeline.
This script loads a sample dataset and runs a simplified version of the pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
RANDOM_STATE = 42
SAMPLE_SIZE = 200
TEST_SIZE = 0.2
N_ESTIMATORS = 100


def load_sample_data(sample_size=SAMPLE_SIZE):
    """
    Create a sample dataset for testing purposes.
    
    Args:
        sample_size (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample DataFrame
    """
    np.random.seed(RANDOM_STATE)
    
    # Network traffic features
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
        'Label': np.random.choice(
            ['BENIGN', 'DoS', 'PortScan', 'Brute Force', 'Web Attack'], 
            sample_size, 
            p=[0.7, 0.1, 0.1, 0.05, 0.05]
        )
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Created sample dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def normalize_column_names(df):
    """
    Normalize column names by removing leading/trailing whitespace,
    replacing spaces with underscores, and converting to lowercase.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with normalized column names
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
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with duplicate columns removed
    """
    columns = df.columns.tolist()
    duplicate_columns = []
    
    for col in columns:
        if col.endswith(('.1', '.2', '.3', '.4', '.5')):
            base_col = col.rsplit('.', 1)[0]
            if base_col in columns and df[col].equals(df[base_col]):
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
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (DataFrame with encoded features, dictionary of label encoders)
    """
    df_encoded = df.copy()
    label_encoders = {}
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'label' in categorical_columns:
        categorical_columns.remove('label')
    
    print(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    # Encode feature columns
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Encode target column
    if 'label' in df.columns:
        le = LabelEncoder()
        df_encoded['label'] = le.fit_transform(df['label'].astype(str))
        label_encoders['label'] = le
        print(f"Encoded label: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df_encoded, label_encoders


def convert_to_float(df):
    """
    Convert all columns (except the target variable) to float.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with numeric columns converted to float
    """
    columns = [col for col in df.columns if col != 'label']
    
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    return df


def main():
    """Main function to test the pipeline."""
    print("Testing Machine Learning Pipeline")
    print("-" * 40)
    
    # Data preprocessing
    df = load_sample_data()
    print(f"Sample data shape: {df.shape}")
    
    df = normalize_column_names(df)
    df = remove_duplicate_columns(df)
    df_encoded, label_encoders = encode_categorical_features(df)
    df_encoded = convert_to_float(df_encoded)
    
    print("Data types after conversion:")
    print(df_encoded.dtypes.head())
    
    # Feature engineering
    X = df_encoded.drop(columns=['label'])
    y = df_encoded['label']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Model training
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = rf.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }
    
    print("\nRandom Forest Results:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print("\nPipeline test completed successfully!")


if __name__ == "__main__":
    main()