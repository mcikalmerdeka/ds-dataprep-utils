# Check Data Information Guide

This document explains the components of the `check_data_information` function found in `utils/preprocessing.py`. This function is designed to provide a comprehensive initial snapshot of your dataset's characteristics, helping you decide on the necessary preprocessing steps.

## Components Breakdown

### 1. Data Type

- **What it is**: The data type of the column (e.g., `int64`, `float64`, `object`, `bool`).
- **Why it's important**:
  - **Memory Usage**: Helps identify if you can downcast types (e.g., `float64` to `float32`) to save memory.
  - **Preprocessing Strategy**: Determines if you need encoders (for `object`), scalers (for `numeric`), or type conversion (e.g., converting "100" string to number).

### 2. Null Values & Null Percentage

- **What it is**: The count and percentage of missing values (`NaN`, `None`) in the column.
- **Why it's important**:
  - **Imputation Strategy**: High missingness (>50%) might suggest dropping the column. Low missingness (<5%) might allow for row deletion. Moderate missingness requires imputation (Mean, Median, Mode, or KNN).
  - **Model Compatibility**: Most ML models (Scikit-Learn) cannot handle nulls and will crash without handling them.

### 3. Zero Values & Zero Percentage

- **What it is**: The count and percentage of rows where the value is exactly `0`.
- **Why it's important**:
  - **Validity Check**: Is `0` a valid value for this feature? (e.g., `Age` = 0 might be an error, but `Debt` = 0 is valid).
  - **Sparsity**: High zero percentage suggests a sparse feature, which might benefit from specific handling or sparse matrices.
  - **Missing Value Indicator**: Sometimes `0` is used as a placeholder for missing values in legacy databases.

### 4. Negative Values & Negative Percentage

- **What it is**: The count and percentage of rows where the value is less than `0`.
- **Why it's important**:
  - **Sanity Check**: Identifies data quality issues. Features like `Price`, `Distance`, `Age`, or `Height` generally should not be negative.
  - **Transformation constraints**: Some transformations (like `Box-Cox` or `Log`) strictly require positive values.

### 5. Empty Strings

- **What it is**: The count of string values that are empty `""` or contain only whitespace `" "`.
- **Why it's important**:
  - **Hidden Nulls**: Standard functions like `pd.isna()` often do not catch empty strings. These behave like missing values but are technically valid strings.
  - **Cleaning**: You need to convert these to `NaN` before imputation or encoding.

### 6. Numeric in Object

- **What it is**: The count of values in an `object` (string) column that can be successfully parsed into numbers.
- **Why it's important**:
  - **Mixed Types Detection**: If you have a column like `["100", "200", "Error", "300"]`, this metric helps you see that most values are numeric.
  - **Fixing Data Types**: Suggests that the column should likely be converted to Numeric after cleaning the non-numeric "garbage" values.

### 7. Duplicated Values

- **What it is**: The count of fully duplicated rows in the dataset (this value is the same for all columns in the summary row).
- **Why it's important**:
  - **Data Leakage**: Duplicates can artificially inflate model performance if they appear in both train and test sets.
  - **Bias**: Repeated records can bias the model towards specific samples.

### 8. Unique Values

- **What it is**: The count of distinct values in the column.
- **Why it's important**:
  - **Categorical vs Continuous**: A low number of unique values implies a Categorical feature (even if it looks numeric, like `Rank` 1-5). A high number implies Continuous data.
  - **Constant Columns**: If Unique Values = 1, the column carries no information and should be dropped.

### 9. Cardinality Ratio

- **What it is**: The ratio of unique values to the total number of rows (`Unique / Total`).
- **Why it's important**:
  - **High Cardinality Detection**: A ratio near `1.0` (e.g., ID columns, Transaction Hashes) usually means the feature is an identifier and not predictive. These should often be dropped.
  - **Encoding Strategy**: Low cardinality suggests One-Hot Encoding. High cardinality suggests Target Encoding or Embedding layers.

### 10. Unique Sample

- **What it is**: A text sample of the first 5 unique values found.
- **Why it's important**:
  - **Quick Context**: Gives you an immediate "feel" for the data content (e.g., seeing `["$100", "$20"]` tells you there are currency symbols to clean).
