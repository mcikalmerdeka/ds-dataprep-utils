# Rapid EDA and Preprocessing Utility Scripts for Data Science Projects

This repository contains a collection of utility scripts and functions designed to accelerate Exploratory Data Analysis (EDA) and Data Preprocessing tasks in Data Science projects. These scripts are optimized for quick usage in environments like Kaggle kernels or local Jupyter notebooks. Initially this repository is created for my personal use, but I think it can be useful for others as well.

## 🚀 Features

The scripts can be found in the `utils` folder. The available scripts are: `preprocessing.py`, `statistics.py`, `visualization.py`.

### 1. Data Inspection & Cleaning

- **`check_data_information`**: Comprehensive summary of data types, null values, duplicates, and unique samples.
- **`handle_missing_values`**: Flexible imputation strategies including:
  - Simple: Mean, Median, Mode
  - Time-series: Forward fill, Backward fill
  - Advanced: KNN Imputation
- **`filter_outliers`**: Detect and filter outliers using IQR or Z-score methods.
- **`drop_columns`**: Safely remove unwanted columns.

### 2. Feature Engineering

- **`feature_scaling`**: Unified interface for multiple scaling methods:
  - Standard Scaler
  - MinMax Scaler
  - Robust Scaler
  - Power Transformer (Yeo-Johnson/Box-Cox)
  - Quantile Transformer
- **`feature_encoding`**: Easy-to-use wrapper for One-Hot and Ordinal encoding, handling unknown values and preserving DataFrames.
- **`change_binary_dtype`**: Toggle binary columns between numerical (0/1) and categorical (No/Yes) formats.

### 3. Statistical Analysis

- **`describe_numerical_combined`**: Detailed stats including skewness, kurtosis, IQR, and CV, with optional grouping (hue).
- **`describe_categorical_combined`**: Frequency analysis, top/bottom categories, and percentages.
- **`describe_date_columns`**: Extract temporal patterns, seasonality, gaps, and ranges from datetime columns.
- **`identify_distribution_types`**: Automatically categorize features as Normal, Skewed, Uniform, etc.

### 4. Visualization

- **`plot_dynamic_hisplots_kdeplots`**: Grid of Histograms/KDEs with hue support.
- **`plot_dynamic_boxplots_violinplots`**: Grid of Box/Violin plots for outlier detection.
- **`plot_dynamic_countplot`**: Grid of count plots for categorical variables.
- **`plot_correlation_heatmap`**: Pearson, Spearman, or Kendall correlation heatmaps.

## 🛠️ Usage

You can simply import the functions you need from the `utils` folder.

```python
# Example Usage
import pandas as pd
from utils.preprocessing import check_data_information, handle_missing_values, feature_scaling

# Load your data
df = pd.read_csv('your_dataset.csv')

# 1. Check Data Info
info = check_data_information(df, df.columns)
print(info)

# 2. Handle Missing Values
df_clean, _ = handle_missing_values(df, columns=['age', 'salary'], strategy='median')

# 3. Scale Features
df_scaled, scaler = feature_scaling(df_clean, columns=['salary'], method='robust')
```

## 🔮 Future Roadmap

This project is actively evolving. I plan to add several more functions and scripts to cover broader data science workflows, including:

- [ ] **Advanced Feature Selection**: Scripts for recursive feature elimination, importance analysis, and multicollinearity checks.
- [ ] **Model Evaluation Utils**: Helper functions for plotting confusion matrices, ROC curves, and comparing model performance metrics.
- [ ] **Time Series Analysis**: Dedicated tools for decomposition, stationarity tests (ADF), and lag analysis.
- [ ] **Text Processing**: Basic NLP preprocessing (cleaning, tokenization, TF-IDF wrappers).
- [ ] **Automated Reporting**: Functions to generate summary HTML or Markdown reports for datasets.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests if you have useful scripts to share!
