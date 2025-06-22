# CSE-CIC-IDS2018 Network Intrusion Detection

This repository contains a machine learning pipeline for network intrusion detection using the CSE-CIC-IDS2018 dataset. The pipeline includes data preprocessing, exploratory data analysis, feature engineering, and various machine learning models for classification.

## Dataset

The CSE-CIC-IDS2018 dataset is a comprehensive network traffic dataset for intrusion detection systems. It contains benign and various attack network flows, including:

- Brute Force attacks
- DoS/DDoS attacks
- Web attacks
- Port scanning
- Botnet activities
- And more

The dataset was created by the Canadian Institute for Cybersecurity (CIC) and the Communications Security Establishment (CSE).

## Repository Structure

- `data/` - Contains the dataset files
- `ml_pipeline_notebook.ipynb` - Jupyter notebook with the complete machine learning pipeline

## Machine Learning Pipeline

The notebook implements a comprehensive machine learning pipeline with the following components:

1. **Data Preprocessing**
   - Normalization of column names
   - Removal of duplicate columns
   - Handling of missing values
   - Encoding of categorical features
   - Outlier detection and removal

2. **Exploratory Data Analysis (EDA)**
   - Statistical analysis of features
   - Correlation analysis
   - Distribution visualization
   - Feature importance analysis

3. **Feature Engineering**
   - Feature selection based on correlation
   - Feature scaling

4. **Machine Learning Models**
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - Support Vector Machine
   - Logistic Regression
   - Neural Networks

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC Curve and AUC
   - Model comparison

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/SergioTGSerra/CSE-CIC-IDS2018.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:
   ```
   jupyter notebook ml_pipeline_notebook.ipynb
   ```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- tensorflow

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Canadian Institute for Cybersecurity (CIC) and Communications Security Establishment (CSE) for providing the dataset
- The original dataset can be found at: https://www.unb.ca/cic/datasets/ids-2018.html