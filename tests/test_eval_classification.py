# Add the parent directory to sys.path to import utils
import sys
import os

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from utils.classification_evals_and_tuning import eval_classification

# Generate synthetic classification data
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, 
                           n_classes=2, random_state=42)

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Running eval_classification...")
metrics = eval_classification(
    model=model,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    y_train=y_train,
    plot_confusion_matrix=False,  # Disable for automated testing
    n_splits=2,
    n_repeats=1
)

print("\nMetrics keys:", metrics.keys())
print("CV metrics keys:", metrics['cv'].keys())
