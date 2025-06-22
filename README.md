# Machine Learning Pipeline for Network Intrusion Detection

This repository contains a comprehensive machine learning pipeline for network intrusion detection using the CSE-CIC-IDS2018 dataset. The pipeline is implemented in a Jupyter notebook and includes data preprocessing, exploratory data analysis, feature engineering, and various machine learning models for classification.

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

Network intrusion detection is a critical component of cybersecurity, aimed at identifying unauthorized access or malicious activities in computer networks. This project implements a machine learning pipeline to detect various types of network intrusions using the CSE-CIC-IDS2018 dataset, which contains benign and attack network traffic data.

## Dataset

The CSE-CIC-IDS2018 dataset is a comprehensive collection of network traffic data that includes various types of attacks, such as DoS, DDoS, brute force, XSS, SQL injection, infiltration, and botnet activities. The dataset is labeled, making it suitable for supervised learning approaches to intrusion detection.

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

The pipeline requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- catboost
- tensorflow
- plotly

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost tensorflow plotly
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/network-intrusion-detection.git
   cd network-intrusion-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   For minimal testing, you can install just the core dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tqdm
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook ml_pipeline_notebook.ipynb
   ```

4. Run the cells in the notebook to execute the pipeline.

### Testing with Sample Data

For quick testing of the pipeline functionality, you can use the provided test script:

```bash
python test_pipeline.py
```

This script creates a synthetic dataset with 200 records and runs a simplified version of the machine learning pipeline to verify that all components work correctly.

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