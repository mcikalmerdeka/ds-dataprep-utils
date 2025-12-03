import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ab_testing import (
    calculate_sample_size,
    perform_t_test,
    perform_mann_whitney_u_test,
    perform_chi_squared_test,
    perform_z_test_proportions,
    check_normality,
    check_variance_homogeneity,
    analyze_ab_test_results
)

def test_sample_size():
    print("\n--- Testing Sample Size Calculation ---")
    n_prop = calculate_sample_size(0.10, 0.02, metric_type='proportion')
    print(f"Sample size for proportion (10% -> 12%): {n_prop}")
    
    n_cont = calculate_sample_size(0.5, 0.05, metric_type='continuous') # Effect size 0.5
    print(f"Sample size for continuous (effect size 0.5): {n_cont}")

def test_statistical_tests():
    print("\n--- Testing Statistical Tests ---")
    
    # Generate sample data
    np.random.seed(42)
    group_a = np.random.normal(100, 10, 100)
    group_b = np.random.normal(105, 10, 100) # Mean difference of 5
    
    # T-Test
    t_res = perform_t_test(group_a, group_b)
    print(f"T-Test: {t_res}")
    print("\n")
    
    # Mann-Whitney U
    mw_res = perform_mann_whitney_u_test(group_a, group_b)
    print(f"Mann-Whitney U: {mw_res}")
    print("\n")
    
    # Normality
    norm_a = check_normality(group_a)
    print(f"Normality Group A: {norm_a}")
    print("\n")
    
    # Variance
    var_res = check_variance_homogeneity(group_a, group_b)
    print(f"Variance Homogeneity: {var_res}")
    print("\n")
    
    # Proportions
    # Group A: 100/1000 (10%), Group B: 130/1000 (13%)
    chi_res = perform_chi_squared_test(100, 1000, 130, 1000)
    print(f"Chi-Squared: {chi_res}")
    print("\n")
    
    z_res = perform_z_test_proportions(100, 1000, 130, 1000)
    print(f"Z-Test: {z_res}")
    print("\n")

def test_high_level_analysis():
    print("\n--- Testing High-Level Analysis ---")
    
    # Create DataFrame
    np.random.seed(42)
    df = pd.DataFrame({
        'group': ['A'] * 100 + ['B'] * 100,
        'revenue': np.concatenate([np.random.normal(100, 10, 100), np.random.normal(105, 10, 100)]),
        'converted': np.concatenate([np.random.binomial(1, 0.1, 100), np.random.binomial(1, 0.15, 100)])
    })
    
    # Analyze Continuous
    print("Analyzing Continuous Metric (Revenue):")
    res_cont = analyze_ab_test_results(df, 'revenue', 'group', metric_type='continuous')
    print(res_cont)
    print("\n")
    
    # Analyze Binary
    print("\nAnalyzing Binary Metric (Conversion):")
    res_bin = analyze_ab_test_results(df, 'converted', 'group', metric_type='binary')
    print(res_bin)
    print("\n")

if __name__ == "__main__":
    try:
        test_sample_size()
        test_statistical_tests()
        test_high_level_analysis()
        print("\nAll manual tests completed successfully.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
