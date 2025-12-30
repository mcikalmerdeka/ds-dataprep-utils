"""
Raw Sample Data Generation Script

This script generates realistic "messy" datasets similar to Kaggle raw data
for testing preprocessing, statistics, visualization, feature selection,
and A/B testing utility functions.

Generated datasets include common data quality issues:
- Missing values (NaN)
- Duplicate rows
- Skewed distributions
- Outliers
- Mixed data types (numeric and categorical)
- Invalid/inconsistent values

Usage:
    python generate_data.py
    
Output:
    - data/classification_data.csv  (Binary classification dataset)
    - data/regression_data.csv      (Regression dataset)
    - data/ab_testing_data.csv      (A/B testing dataset)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def add_missing_values(df: pd.DataFrame, missing_rate: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """Add random missing values to a DataFrame."""
    np.random.seed(random_state)
    df = df.copy()
    
    for col in df.columns:
        mask = np.random.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    
    return df


def add_outliers(series: pd.Series, outlier_rate: float = 0.05, multiplier: float = 5, random_state: int = 42) -> pd.Series:
    """Add outliers to a numeric series."""
    np.random.seed(random_state)
    series = series.copy()
    
    n_outliers = int(len(series) * outlier_rate)
    outlier_indices = np.random.choice(series.index, size=n_outliers, replace=False)
    
    std = series.std()
    mean = series.mean()
    
    # Add extreme values (both high and low)
    for idx in outlier_indices:
        if np.random.random() > 0.5:
            series.loc[idx] = mean + multiplier * std * (1 + np.random.random())
        else:
            series.loc[idx] = mean - multiplier * std * (1 + np.random.random())
    
    return series


def create_skewed_distribution(n_samples: int, skew_direction: str = 'right', random_state: int = 42) -> np.ndarray:
    """Create a skewed distribution (right-skewed or left-skewed)."""
    np.random.seed(random_state)
    
    if skew_direction == 'right':
        # Right-skewed (positive skew) - e.g., income, prices
        return np.random.exponential(scale=1000, size=n_samples)
    elif skew_direction == 'left':
        # Left-skewed (negative skew) - e.g., age at retirement
        return 100 - np.random.exponential(scale=20, size=n_samples)
    else:
        # Normal distribution
        return np.random.normal(loc=50, scale=15, size=n_samples)


def generate_classification_data(
    n_samples: int = 1000,
    missing_rate: float = 0.08,
    duplicate_rate: float = 0.05,
    random_state: int = 42,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generate a realistic binary classification dataset with data quality issues.
    
    Scenario: Customer Churn Prediction
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    missing_rate : float
        Proportion of missing values to introduce
    duplicate_rate : float
        Proportion of duplicate rows to add
    random_state : int
        Random seed for reproducibility
    save_path : str, optional
        Path to save the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(random_state)
    
    # Generate customer data
    data = {
        # Customer ID (some duplicates)
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        
        # Age - slightly right-skewed with some outliers
        'age': np.clip(np.random.normal(40, 15, n_samples), 18, 90).astype(int),
        
        # Gender - categorical
        'gender': np.random.choice(['Male', 'Female', 'Other', 'male', 'FEMALE'], n_samples, 
                                    p=[0.45, 0.45, 0.02, 0.04, 0.04]),  # Inconsistent casing
        
        # Tenure (months) - right-skewed
        'tenure_months': np.abs(np.random.exponential(24, n_samples)).astype(int),
        
        # Monthly charges - normal with outliers
        'monthly_charges': np.random.normal(70, 30, n_samples),
        
        # Total charges - derived but with noise
        'total_charges': None,  # Will be calculated
        
        # Contract type - categorical
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year', 'monthly', np.nan], 
                                          n_samples, p=[0.5, 0.25, 0.2, 0.04, 0.01]),
        
        # Payment method - categorical with some missing
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 
                                            'Mailed Check', None], n_samples, 
                                           p=[0.3, 0.25, 0.25, 0.15, 0.05]),
        
        # Number of support tickets - count data, right-skewed
        'support_tickets': np.random.poisson(2, n_samples),
        
        # Account balance - can be negative (credit) or positive
        'account_balance': np.random.normal(0, 100, n_samples),
        
        # Internet service - categorical
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No', 'dsl', 'fiber'], n_samples,
                                             p=[0.3, 0.4, 0.2, 0.05, 0.05]),
        
        # Online security - binary with some invalid values
        'online_security': np.random.choice(['Yes', 'No', 'yes', 'NO', '1', '0', 'True'], n_samples,
                                            p=[0.35, 0.35, 0.1, 0.1, 0.03, 0.04, 0.03]),
        
        # Customer satisfaction score (1-5) - with outliers
        'satisfaction_score': np.clip(np.random.normal(3.5, 1, n_samples), 1, 5).round(1),
        
        # Last login days ago - right-skewed
        'days_since_last_login': np.abs(np.random.exponential(15, n_samples)).astype(int),
        
        # Number of products - count
        'num_products': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        
        # Referral source - categorical with high cardinality
        'referral_source': np.random.choice(['Google', 'Facebook', 'Friend', 'TV Ad', 'Billboard',
                                             'Instagram', 'Twitter', 'Email', 'Other', np.nan], n_samples,
                                            p=[0.2, 0.15, 0.15, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure zero values in tenure (new customers)
    zero_tenure_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[zero_tenure_idx, 'tenure_months'] = 0
    
    # Calculate total charges with some noise and issues
    df['total_charges'] = (df['monthly_charges'] * df['tenure_months'] * 
                           np.random.uniform(0.95, 1.05, n_samples))
    df.loc[df['tenure_months'] == 0, 'total_charges'] = 0
    
    # Add some negative values (data entry errors)
    error_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[error_indices, 'monthly_charges'] = -df.loc[error_indices, 'monthly_charges']
    
    # Add outliers to specific columns
    df['monthly_charges'] = add_outliers(df['monthly_charges'], outlier_rate=0.03, random_state=random_state)
    df['age'] = add_outliers(df['age'].astype(float), outlier_rate=0.02, multiplier=3, random_state=random_state+1)
    
    # --- Numeric in Object & Empty strings injection (Classification) ---
    
    # 1. Total Charges: Make it an object column with mixed types (Numeric-in-Object)
    # Convert to string first
    df['total_charges'] = df['total_charges'].astype(str)
    
    # Inject Empty Strings / Whitespace (simulating missing data entry)
    empty_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[empty_indices, 'total_charges'] = np.random.choice(["", " ", "  "], size=len(empty_indices))
    
    # Inject some text garbage to ensure it's treated as object (e.g. "Payment Pending")
    text_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[text_indices, 'total_charges'] = "Pending"
    
    # 2. Payment Method: Inject specific Empty Strings (Categorical)
    # We replace some valid values with empty strings
    empty_payment_idx = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[empty_payment_idx, 'payment_method'] = np.random.choice(["", " "], size=len(empty_payment_idx))
    
    # Generate target variable (churn) - imbalanced
    churn_prob = 0.2 + 0.3 * (df['tenure_months'] < 12).astype(float)
    churn_prob += 0.1 * (df['support_tickets'] > 3).astype(float)
    churn_prob += 0.1 * (df['satisfaction_score'] < 3).astype(float)
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, n_samples), 0, 1)
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Add some string representations of target
    target_map = {0: 'No', 1: 'Yes'}
    df['churn_label'] = df['churn'].map(target_map)
    
    # Add missing values
    cols_to_add_missing = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                           'satisfaction_score', 'days_since_last_login']
    for col in cols_to_add_missing:
        mask = np.random.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    
    # Add duplicate rows
    n_duplicates = int(n_samples * duplicate_rate)
    duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=True)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save to CSV if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ Classification data saved to: {save_path}")
    
    return df


def generate_regression_data(
    n_samples: int = 1000,
    missing_rate: float = 0.08,
    duplicate_rate: float = 0.05,
    random_state: int = 42,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generate a realistic regression dataset with data quality issues.
    
    Scenario: House Price Prediction
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    missing_rate : float
        Proportion of missing values to introduce
    duplicate_rate : float
        Proportion of duplicate rows to add
    random_state : int
        Random seed for reproducibility
    save_path : str, optional
        Path to save the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(random_state)
    
    # Generate house data
    data = {
        # Property ID
        'property_id': [f'PROP_{i:05d}' for i in range(n_samples)],
        
        # Square footage - right-skewed
        'sqft_living': create_skewed_distribution(n_samples, 'right', random_state) + 1000,
        
        # Lot size - very right-skewed
        'sqft_lot': create_skewed_distribution(n_samples, 'right', random_state + 1) * 5 + 2000,
        
        # Number of bedrooms - count with outliers
        'bedrooms': np.clip(np.random.poisson(3, n_samples), 1, 10),
        
        # Number of bathrooms - with decimal values
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], n_samples,
                                      p=[0.1, 0.15, 0.25, 0.2, 0.15, 0.05, 0.05, 0.03, 0.02]),
        
        # Number of floors
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.3, 0.1, 0.4, 0.1, 0.1]),
        
        # Year built - left-skewed (more recent houses in dataset)
        'year_built': np.clip(2024 - np.abs(np.random.exponential(30, n_samples)), 1900, 2024).astype(int),
        
        # Year renovated (0 if never renovated) - many zeros
        'year_renovated': np.where(np.random.random(n_samples) < 0.7, 0,
                                   np.clip(2024 - np.abs(np.random.exponential(15, n_samples)), 1980, 2024).astype(int)),
        
        # Condition (1-5) - ordinal
        'condition': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.35, 0.35, 0.15]),
        
        # Grade (1-13) - quality rating, normally distributed
        'grade': np.clip(np.random.normal(7, 2, n_samples), 1, 13).astype(int),
        
        # Waterfront - binary, rare
        'waterfront': np.random.choice([0, 1, 'Yes', 'No', 'yes'], n_samples, 
                                       p=[0.85, 0.05, 0.03, 0.05, 0.02]),
        
        # View score (0-4)
        'view': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        
        # City/Location - categorical
        'city': np.random.choice(['Seattle', 'Bellevue', 'Redmond', 'Kirkland', 'Tacoma',
                                  'seattle', 'BELLEVUE', 'Other'], n_samples,
                                 p=[0.3, 0.15, 0.12, 0.1, 0.15, 0.05, 0.05, 0.08]),
        
        # Zipcode - high cardinality categorical
        'zipcode': np.random.randint(98001, 98200, n_samples).astype(str),
        
        # Latitude and Longitude
        'latitude': np.random.uniform(47.1, 47.8, n_samples),
        'longitude': np.random.uniform(-122.5, -121.5, n_samples),
        
        # Distance to downtown (miles) - right-skewed
        'distance_to_downtown': np.abs(np.random.exponential(10, n_samples)),
        
        # School rating (1-10)
        'school_rating': np.clip(np.random.normal(7, 1.5, n_samples), 1, 10).round(1),
        
        # Crime rate (per 1000) - right-skewed
        'crime_rate': np.abs(np.random.exponential(3, n_samples)),
        
        # Garage - binary/categorical
        'has_garage': np.random.choice(['Yes', 'No', 1, 0, 'TRUE', 'false'], n_samples,
                                       p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05]),
        
        # Basement square footage - many zeros
        'sqft_basement': np.where(np.random.random(n_samples) < 0.4, 0,
                                  np.abs(np.random.normal(500, 200, n_samples))),
    }
    
    df = pd.DataFrame(data)
    
    # Add some extreme outliers to bedrooms
    outlier_idx = np.random.choice(df.index, size=int(n_samples * 0.01), replace=False)
    df.loc[outlier_idx, 'bedrooms'] = np.random.choice([15, 20, 33, 0], len(outlier_idx))
    
    # Add outliers to sqft
    df['sqft_living'] = add_outliers(df['sqft_living'], outlier_rate=0.03, multiplier=4, random_state=random_state)
    
    # --- Numeric in Object & Empty strings injection (Regression) ---
    
    # 1. Sqft Basement: Make it an object column with mixed types (Numeric-in-Object)
    # Convert to string
    df['sqft_basement'] = df['sqft_basement'].astype(str)
    
    # Inject Empty Strings / Whitespace
    empty_bsmt_idx = np.random.choice(df.index, size=int(n_samples * 0.06), replace=False)
    df.loc[empty_bsmt_idx, 'sqft_basement'] = np.random.choice(["", " ", "  "], size=len(empty_bsmt_idx))
    
    # Inject text values (e.g., "None", "TBD")
    text_bsmt_idx = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[text_bsmt_idx, 'sqft_basement'] = np.random.choice(["None", "TBA", "Check Plan"], size=len(text_bsmt_idx))
    
    # 2. City: Inject Empty Strings
    empty_city_idx = np.random.choice(df.index, size=int(n_samples * 0.04), replace=False)
    df.loc[empty_city_idx, 'city'] = np.random.choice(["", " "], size=len(empty_city_idx))
    
    # 3. Inject Negative Values (Anomalies in Lot Size - Physically impossible)
    neg_lot_idx = np.random.choice(df.index, size=int(n_samples * 0.01), replace=False)
    df.loc[neg_lot_idx, 'sqft_lot'] = -1 * df.loc[neg_lot_idx, 'sqft_lot']
    
    # Generate target variable (price) - based on features with noise
    base_price = (
        df['sqft_living'].fillna(df['sqft_living'].median()) * 150 +
        df['bedrooms'].fillna(3) * 15000 +
        df['bathrooms'].fillna(2) * 20000 +
        df['grade'].fillna(7) * 25000 +
        (2024 - df['year_built'].fillna(1990)) * -500 +
        df['school_rating'].fillna(7) * 10000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    # Add waterfront premium (mixed types need handling)
    waterfront_binary = df['waterfront'].isin([1, 'Yes', 'yes']).astype(int)
    base_price += waterfront_binary * 200000
    
    df['price'] = np.clip(base_price, 50000, 5000000)
    
    # Add some very skewed price outliers (luxury homes)
    luxury_idx = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[luxury_idx, 'price'] = np.random.uniform(2000000, 10000000, len(luxury_idx))
    
    # Add missing values to specific columns
    cols_to_add_missing = ['sqft_living', 'sqft_lot', 'year_built', 'year_renovated', 
                           'grade', 'school_rating', 'crime_rate', 'sqft_basement', 'price']
    for col in cols_to_add_missing:
        mask = np.random.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    
    # Add duplicate rows
    n_duplicates = int(n_samples * duplicate_rate)
    duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=True)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save to CSV if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ Regression data saved to: {save_path}")
    
    return df


def generate_ab_testing_data(
    n_samples: int = 2000,
    effect_size: float = 0.05,
    random_state: int = 42,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generate a dataset for A/B testing analysis.
    
    Scenario: Website Conversion Rate Experiment
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    effect_size : float
        Expected effect size (difference in conversion rate)
    random_state : int
        Random seed for reproducibility
    save_path : str, optional
        Path to save the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(random_state)
    
    # Split into control and treatment groups
    n_control = n_samples // 2
    n_treatment = n_samples - n_control
    
    # Base conversion rate
    base_conversion = 0.10
    
    # Control group
    control_data = {
        'user_id': [f'USER_{i:05d}' for i in range(n_control)],
        'group': 'control',
        'converted': np.random.binomial(1, base_conversion, n_control),
        'time_on_page': np.abs(np.random.exponential(120, n_control)),  # seconds
        'pages_viewed': np.random.poisson(4, n_control),
        'device': np.random.choice(['Desktop', 'Mobile', 'Tablet'], n_control, p=[0.5, 0.4, 0.1]),
        'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge', 'Other'], n_control,
                                    p=[0.55, 0.15, 0.15, 0.1, 0.05]),
        'country': np.random.choice(['US', 'UK', 'Canada', 'Australia', 'Germany', 'Other'], n_control,
                                    p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.15]),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_control,
                                      p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        'returning_user': np.random.choice([True, False], n_control, p=[0.3, 0.7]),
    }
    
    # Treatment group (with effect)
    treatment_data = {
        'user_id': [f'USER_{i:05d}' for i in range(n_control, n_samples)],
        'group': 'treatment',
        'converted': np.random.binomial(1, base_conversion + effect_size, n_treatment),
        'time_on_page': np.abs(np.random.exponential(130, n_treatment)),  # slightly higher
        'pages_viewed': np.random.poisson(4.5, n_treatment),  # slightly higher
        'device': np.random.choice(['Desktop', 'Mobile', 'Tablet'], n_treatment, p=[0.5, 0.4, 0.1]),
        'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge', 'Other'], n_treatment,
                                    p=[0.55, 0.15, 0.15, 0.1, 0.05]),
        'country': np.random.choice(['US', 'UK', 'Canada', 'Australia', 'Germany', 'Other'], n_treatment,
                                    p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.15]),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_treatment,
                                      p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        'returning_user': np.random.choice([True, False], n_treatment, p=[0.3, 0.7]),
    }
    
    # Combine
    control_df = pd.DataFrame(control_data)
    treatment_df = pd.DataFrame(treatment_data)
    df = pd.concat([control_df, treatment_df], ignore_index=True)
    
    # Add revenue (only for converted users)
    df['revenue'] = 0.0
    converted_mask = df['converted'] == 1
    df.loc[converted_mask, 'revenue'] = np.random.exponential(50, converted_mask.sum())
    
    # Add timestamp
    start_date = pd.Timestamp('2024-01-01')
    df['timestamp'] = start_date + pd.to_timedelta(np.random.randint(0, 30*24*60, n_samples), unit='m')
    
    # Add some missing values
    missing_idx = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[missing_idx, 'time_on_page'] = np.nan
    
    missing_idx2 = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_idx2, 'age_group'] = np.nan
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save to CSV if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ A/B testing data saved to: {save_path}")
    
    return df


if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = Path(__file__).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Raw Sample Data Generation for Utils Testing")
    print("=" * 70)
    
    # Generate Classification Data
    print("\n📊 Generating Classification Dataset (Customer Churn)...")
    clf_path = data_dir / "classification_data.csv"
    clf_df = generate_classification_data(n_samples=1000, save_path=str(clf_path))
    print(f"   Shape: {clf_df.shape}")
    print(f"   Missing values: {clf_df.isnull().sum().sum()}")
    print(f"   Duplicates: {clf_df.duplicated().sum()}")
    print(f"   Target distribution:\n{clf_df['churn'].value_counts()}")
    
    # Generate Regression Data
    print("\n📈 Generating Regression Dataset (House Prices)...")
    reg_path = data_dir / "regression_data.csv"
    reg_df = generate_regression_data(n_samples=1000, save_path=str(reg_path))
    print(f"   Shape: {reg_df.shape}")
    print(f"   Missing values: {reg_df.isnull().sum().sum()}")
    print(f"   Duplicates: {reg_df.duplicated().sum()}")
    print(f"   Target range: ${reg_df['price'].min():,.0f} - ${reg_df['price'].max():,.0f}")
    
    # Generate A/B Testing Data
    print("\n🔬 Generating A/B Testing Dataset (Website Conversion)...")
    ab_path = data_dir / "ab_testing_data.csv"
    ab_df = generate_ab_testing_data(n_samples=2000, save_path=str(ab_path))
    print(f"   Shape: {ab_df.shape}")
    print(f"   Groups: {ab_df['group'].value_counts().to_dict()}")
    print(f"   Conversion rates:")
    for group in ['control', 'treatment']:
        rate = ab_df[ab_df['group'] == group]['converted'].mean()
        print(f"      {group}: {rate:.2%}")
    
    print("\n" + "=" * 70)
    print("✅ All datasets generated successfully!")
    print("=" * 70)
    
    print("\n📁 Files created:")
    print(f"   • {clf_path}")
    print(f"   • {reg_path}")
    print(f"   • {ab_path}")
    
    print("\n📝 Data Quality Issues Included:")
    print("   • Missing values (NaN)")
    print("   • Duplicate rows")
    print("   • Inconsistent categorical values (e.g., 'Male', 'male', 'MALE')")
    print("   • Outliers in numeric columns")
    print("   • Skewed distributions")
    print("   • Mixed data types (e.g., 'Yes', 'No', 1, 0 in same column)")
    print("   • Numeric-in-Object (e.g., numbers stored as strings mixed with 'Pending')")
    print("   • Empty strings and whitespaces")
    print("   • Invalid/negative values where inappropriate")
    print("   • High cardinality categorical features")
    
    print("\n💡 Use these datasets to test your utility scripts!")
