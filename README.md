# Machine Learning Pipeline for Network Intrusion Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-yellow)](https://www.tensorflow.org/)

A comprehensive machine learning pipeline for network intrusion detection using the CSE-CIC-IDS2018 dataset. This project implements a complete workflow from data preprocessing to model evaluation, designed to detect various types of network attacks.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Workflow](#pipeline-workflow)
- [Requirements](#requirements)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Machine Learning Models](#machine-learning-models)
  - [Model Evaluation](#model-evaluation)
  - [Model Usage](#model-usage)

## Overview

Network intrusion detection is a critical component of cybersecurity that identifies unauthorized access or malicious activities in computer networks. This project provides:

- A complete machine learning pipeline for network traffic analysis
- Multiple classification models to detect various attack types
- Comprehensive data preprocessing and feature engineering techniques
- Detailed model evaluation and comparison metrics

## Dataset

The [CSE-CIC-IDS2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) contains network traffic data with various attack types:

- DoS (Denial of Service)
- DDoS (Distributed Denial of Service)
- Brute Force attacks
- Web attacks (XSS, SQL Injection)
- Infiltration and botnet activities

The dataset is labeled, making it suitable for supervised learning approaches to intrusion detection.

## Pipeline Workflow

The machine learning pipeline follows these main steps:

1. **Data Preprocessing and Transformation**
   - Load dataset from CSV files
   - Inspect data types and count non-null values
   - Normalize column names
   - Remove duplicate columns
   - Convert categorical features using Label Encoding
   - Convert all data types to float
   - Replace infinite values with NaN
   - Handle missing values by imputing with column mean
   - Replace nonsensical negative values with median
   - Detect and remove outliers using Isolation Forest
   - Scale numerical features using StandardScaler

2. **Exploratory Data Analysis (EDA)**
   - Compute and visualize descriptive statistics
   - Analyze and visualize data distributions
   - Perform correlation and relationship analysis
   - Visualize outliers using boxplots

3. **Feature Engineering**
   - Select features based on correlation with the target
   - Split dataset into features (X) and target (y)

4. **Machine Learning Phase**
   - Split data into training (80%) and testing (20%) sets
   - Apply K-Means clustering and determine optimal k
   - Perform dimensionality reduction with PCA
   - Visualize clusters in 2D and 3D
   - Train and evaluate regression models
   - Train and evaluate multiple classification models

5. **Model Evaluation**
   - Compute and compare accuracy, precision, recall, and F1 score
   - Visualize confusion matrices
   - Plot ROC curves and compute AUC scores
   - Compare models using various metrics
   - Visualize decision tree structure

6. **Model Usage**
   - Implement a prediction function for new data
   - Demonstrate how to use the trained models

## Requirements

This project requires Python 3.8+ and the following libraries:

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms

### Visualization
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations

### Advanced ML Models
- **xgboost** - Gradient boosting framework
- **lightgbm** - Gradient boosting framework
- **catboost** - Gradient boosting framework
- **tensorflow** - Deep learning framework

All dependencies are specified in the `requirements.txt` file.

## Usage

### Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/SergioTGSerra/CSE-CIC-IDS2018.git
   cd CSE-CIC-IDS2018
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the test pipeline with sample data:
   ```bash
   python test_pipeline.py
   ```

### Full Pipeline

1. Download the [CSE-CIC-IDS2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) and place the CSV files in the `data` directory.

2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook ml_pipeline_notebook_v3.ipynb
   ```

### Memory-Efficient Version

For systems with limited memory, use the memory-efficient version of the notebook:
```bash
jupyter notebook ml_pipeline_notebook_v2.ipynb
```

This version loads and processes data in chunks to reduce memory usage.

## Pipeline Components

### Data Preprocessing

The data preprocessing phase includes several steps to clean and prepare the data for analysis:

- **Loading Data**: The pipeline loads CSV files from the specified directory and combines them into a single DataFrame.
- **Data Cleaning**: It handles missing values, removes duplicates, and normalizes column names.
- **Feature Transformation**: Categorical features are encoded, and numerical features are scaled.
- **Outlier Detection**: The Isolation Forest algorithm is used to detect and remove outliers.

### Exploratory Data Analysis

The EDA phase provides insights into the dataset:

- **Descriptive Statistics**: Computes and visualizes minimum, maximum, mean, median, mode, percentiles, and standard deviation.
- **Distribution Analysis**: Visualizes and analyzes normal, binomial, Poisson, and Student's t-distributions.
- **Correlation Analysis**: Creates correlation matrix heatmaps and scatter plot matrices.
- **Outlier Visualization**: Uses boxplots to visualize the distribution of features and identify outliers.

### Feature Engineering

The feature engineering phase prepares the data for machine learning:

- **Feature Selection**: Selects features based on their correlation with the target variable.
- **Data Splitting**: Splits the dataset into features (X) and target (y).

### Machine Learning Models

The pipeline implements and evaluates various machine learning models:

- **Clustering**: K-Means clustering with optimal k determination.
- **Dimensionality Reduction**: Principal Component Analysis (PCA).
- **Regression**: Linear Regression.
- **Classification**: Multiple classification models, including:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - AdaBoost
  - XGBoost
  - CatBoost
  - LightGBM
  - Recurrent Neural Network (RNN)

### Model Evaluation

The model evaluation phase assesses the performance of the trained models:

- **Metrics**: Computes accuracy, precision, recall, and F1 score.
- **Visualizations**: Creates confusion matrices, ROC curves, and model comparison plots.
- **Decision Tree Visualization**: Visualizes the structure of the decision tree model.

### Model Usage

The model usage phase demonstrates how to use the trained models for prediction:

- **Prediction Function**: Implements a function to predict the class of a single data sample.
- **Example Usage**: Provides an example of how to use the prediction function with new data.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The CSE-CIC-IDS2018 dataset is provided by the Canadian Institute for Cybersecurity (CIC) and the Communications Security Establishment (CSE).
- This project is inspired by the need for effective network intrusion detection systems in cybersecurity.