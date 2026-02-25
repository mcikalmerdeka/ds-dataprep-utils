# Import metrics score that will be evaluated
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    get_scorer,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║            Functions for Machine Learning Evaluation                             ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

# Function for evaluation score calculation and display confusion matrix
def eval_classification(
    model,
    X_test,
    y_test,
    X_train,
    y_train,
    custom_metrics=None,
    plot_confusion_matrix=True,
    plot_roc_curve=True,
    plot_pr_curve=True,
    plot_calibration_curve=True,
    plot_threshold_analysis=True,
    validate_confusion_matrix=False,
    n_splits=5,
    n_repeats=3,
    random_state=42
) -> dict:
    """
    Comprehensively evaluate a classification model using multiple metrics 
    and cross-validation techniques.

    This function provides a robust evaluation of a classification model by:
    1. Performing single-run predictions
    2. Conducting cross-validated performance assessment
    3. Computing multiple performance metrics
    4. Optionally visualizing the cross-validated confusion matrix

    Parameters:
    -----------
    model : sklearn estimator
        The classification model to be evaluated
    X_test : array-like
        Test feature dataset
    y_test : array-like
        Test target labels
    X_train : array-like
        Training feature dataset
    y_train : array-like
        Training target labels
    custom_metrics : dict, optional (default=None)
        Additional custom metric functions to evaluate
        Format: {'metric_name': metric_function}
    plot_confusion_matrix : bool, optional (default=True)
        Whether to display the confusion matrix
    plot_roc_curve : bool, optional (default=True)
        Whether to display the cross-validated ROC curve with mean AUC
    plot_pr_curve : bool, optional (default=True)
        Whether to display the cross-validated Precision-Recall curve with Average Precision
    plot_calibration_curve : bool, optional (default=True)
        Whether to display the cross-validated Calibration curve (reliability diagram)
    plot_threshold_analysis : bool, optional (default=True)
        Whether to display the Decision Threshold Analysis curve
    validate_confusion_matrix : bool, optional (default=False)
        Whether to use the normal confusion matrix or validated confusion matrix 
    n_splits : int, optional (default=5)
        Number of splits for cross-validation
    n_repeats : int, optional (default=3)
        Number of times to repeat cross-validation
    random_state : int, optional (default=42)
        Seed for reproducibility of cross-validation splits

    Example:
    --------
    >>> metrics = eval_classification(
    >>>     model=model,
    >>>     X_test=X_test,
    >>>     y_test=y_test,
    >>>     X_train=X_train,
    >>>     y_train=y_train
    >>> )
    >>> print(metrics)

    Returns:
    --------
    dict
        Comprehensive dictionary of model performance metrics including:
        - Cross-validated metrics
        - Single-run metrics
        - Cross-validated confusion matrix
    
    Raises:
    -------
    Exception
        If any error occurs during model evaluation
    """
    try:
        # Utility function to convert input to numpy array
        def to_numpy(arr):
            """
            Convert pandas DataFrame/Series to numpy array if necessary.
            
            Parameters:
            -----------
            arr : array-like
                Input array to be converted
            
            Returns:
            --------
            numpy.ndarray
                Converted numpy array
            """
            if isinstance(arr, (pd.DataFrame, pd.Series)):
                return arr.to_numpy()
            return arr

        # Convert all input data to numpy arrays
        X_test = to_numpy(X_test)
        y_test = to_numpy(y_test)
        X_train = to_numpy(X_train)
        y_train = to_numpy(y_train)

        # Initialize metrics dictionary to store all evaluation results
        metrics = {}

        # Perform single-run predictions on test and train datasets
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Get probability predictions for metrics like ROC AUC
        y_pred_proba_test = model.predict_proba(X_test)
        y_pred_proba_train = model.predict_proba(X_train)

        # Setup cross-validation strategy
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )

        if validate_confusion_matrix:

            # List to store confusion matrices from each fold
            conf_matrix_list = []

            # Perform cross-validation manually to collect confusion matrices
            for train_index, val_index in cv.split(X_train, y_train):
                # Split data for this fold
                X_train_fold = X_train[train_index]
                X_val_fold = X_train[val_index]
                y_train_fold = y_train[train_index]
                y_val_fold = y_train[val_index]

                # Clone model to prevent modification of original model
                model_fold = clone(model)
                
                # Fit model on training fold and predict on validation fold
                model_fold.fit(X_train_fold, y_train_fold)
                fold_preds = model_fold.predict(X_val_fold)

                # Compute and store confusion matrix for this fold
                cv_confusion_matrix = confusion_matrix(y_val_fold, fold_preds)
                # print(f"individual element : {cv_confusion_matrix}")
                
                conf_matrix_list.append(cv_confusion_matrix)
                # print(f"appended result : {np.vstack(conf_matrix_list)}")

            # Compute average confusion matrix across all folds
            mean_conf_matrix = np.rint(np.mean(conf_matrix_list, axis=0)).astype(int)
            # print(f"mean of all : {mean_conf_matrix}")

            metrics['cv_confusion_matrix'] = mean_conf_matrix
        
        else:
            normal_conf_matrix = confusion_matrix(y_test, y_pred_test)
            metrics['confusion_matrix'] = normal_conf_matrix

        # Define standard metrics to evaluate
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        # Define metric functions explicitly (avoid fragile globals() lookup)
        metric_funcs = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
        }

        # Perform cross-validation with multiple metrics
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1  # Use all available CPU cores
        )

        # Store cross-validated metrics
        metrics['cv'] = {}

        # Process each metric from cross-validation
        for metric in scoring.keys():
            test_scores = cv_scores[f'test_{metric}']
            train_scores = cv_scores[f'train_{metric}']

            # Store mean and standard deviation of cross-validated scores
            metrics['cv'][metric] = {
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std()
            }

            # Store single-run metrics
            if metric == 'roc_auc':
                metrics[metric] = {
                    'test': roc_auc_score(y_test, y_pred_proba_test[:, 1]),
                    'train': roc_auc_score(y_train, y_pred_proba_train[:, 1])
                }
            else:
                metrics[metric] = {
                    'test': metric_funcs[metric](y_test, y_pred_test),
                    'train': metric_funcs[metric](y_train, y_pred_train)
                }

        # Add custom metrics if provided
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                metrics[metric_name] = {
                    'test': metric_func(y_test, y_pred_test),
                    'train': metric_func(y_train, y_pred_train)
                }

        # Print results
        print(f"Performance Metrics for {model.__class__.__name__}:")
        print("\nCross-Validated Metrics (mean ± std):")
        for metric in scoring.keys():
            print(f"\n{metric.upper()}:")
            print(f"  Test:  {metrics['cv'][metric]['test_mean']*100:.2f} ± {metrics['cv'][metric]['test_std']*100:.2f}")
            print(f"  Train: {metrics['cv'][metric]['train_mean']*100:.2f} ± {metrics['cv'][metric]['train_std']*100:.2f}")

        print("\nSingle Run Metrics:")
        for metric in scoring.keys():
            print(f"\n{metric.upper()}:")
            print(f"  Test:  {metrics[metric]['test']*100:.2f}")
            print(f"  Train: {metrics[metric]['train']*100:.2f}")

        # Plot confusion matrix if requested
        if plot_confusion_matrix:

            if validate_confusion_matrix:
                conf_matrix_values = mean_conf_matrix
                title_prefix = "Cross-Validation"
            else:
                conf_matrix_values = normal_conf_matrix
                title_prefix = "Test Set"

            plt.figure(figsize=(8, 6))
            ax = plt.subplot()

            # Create annotation labels with TN/FN/FP/TP
            labels = np.array([
                [f'TN\n{conf_matrix_values[0, 0]}', f'FP\n{conf_matrix_values[0, 1]}'],
                [f'FN\n{conf_matrix_values[1, 0]}', f'TP\n{conf_matrix_values[1, 1]}']
            ])

            sns.heatmap(
                conf_matrix_values,
                ax=ax,
                annot=labels,
                fmt='',  # Empty format since we're using string annotations
                cmap='Blues',
                xticklabels=['False (0)', 'True (1)'],
                yticklabels=['False (0)', 'True (1)'],
                annot_kws={'fontsize': 12, 'fontweight': 'bold'}
            )

            # Add labels and title
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('Actual Label')
            ax.set_title(f'{title_prefix} Confusion Matrix - {model.__class__.__name__}')
            plt.tight_layout()
            plt.show()

        # Plot ROC curve if requested (using cross-validated predictions)
        if plot_roc_curve:
            from sklearn.metrics import auc
            
            # Lists to store ROC data from each fold
            tpr_list = []
            auc_list = []
            mean_fpr = np.linspace(0, 1, 100)
            
            # Perform cross-validation to collect ROC curves from each fold
            for train_index, val_index in cv.split(X_train, y_train):
                # Clone and fit model on this fold
                model_fold = clone(model)
                model_fold.fit(X_train[train_index], y_train[train_index])
                
                # Get probabilities for the positive class
                y_proba = model_fold.predict_proba(X_train[val_index])[:, 1]
                y_true = y_train[val_index]
                
                # Calculate ROC curve for this fold
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Interpolate TPR at common FPR points
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tpr_list.append(interp_tpr)
                auc_list.append(roc_auc)
            
            # Calculate mean and std
            mean_tpr = np.mean(tpr_list, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(auc_list)
            std_auc = np.std(auc_list)
            std_tpr = np.std(tpr_list, axis=0)
            tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
            
            # Get CV metrics for display
            cv_auc_mean = metrics['cv']['roc_auc']['test_mean']
            cv_auc_std = metrics['cv']['roc_auc']['test_std']
            
            # Plot ROC curve
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot mean ROC curve
            ax.plot(mean_fpr, mean_tpr, color='darkorange', linewidth=2.5,
                   label=f'Mean ROC (CV AUC = {cv_auc_mean:.3f} ± {cv_auc_std:.3f})',
                   zorder=3)
            
            # Plot standard deviation band
            ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='darkorange', alpha=0.2,
                           label=f'±1 std. dev.')
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8,
                   label='Chance level (AUC = 0.500)')
            
            # Configure plot
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
            ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
            ax.set_title(f'Cross-Validated ROC Curve - {model.__class__.__name__}\n'
                        f'({n_splits}-fold × {n_repeats} repeats)', fontsize=14)
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

        # Plot Precision-Recall Curve if requested
        if plot_pr_curve:
            from sklearn.metrics import auc as sklearn_auc
            
            # Lists to store PR data from each fold
            precision_list = []
            ap_list = []
            mean_recall = np.linspace(0, 1, 100)
            
            # Perform cross-validation to collect PR curves from each fold
            for train_index, val_index in cv.split(X_train, y_train):
                model_fold = clone(model)
                model_fold.fit(X_train[train_index], y_train[train_index])
                
                y_proba = model_fold.predict_proba(X_train[val_index])[:, 1]
                y_true = y_train[val_index]
                
                # Calculate PR curve for this fold
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                avg_precision = average_precision_score(y_true, y_proba)
                
                # Interpolate precision at common recall points
                interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                interp_precision[0] = 1.0
                precision_list.append(interp_precision)
                ap_list.append(avg_precision)
            
            # Calculate mean and std
            mean_precision = np.mean(precision_list, axis=0)
            mean_precision[-1] = 0.0
            mean_ap = np.mean(ap_list)
            std_ap = np.std(ap_list)
            std_precision = np.std(precision_list, axis=0)
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            
            # Get baseline (random classifier)
            baseline = np.mean(y_train)
            
            # Plot PR curve
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot mean PR curve
            ax.plot(mean_recall, mean_precision, color='darkblue', linewidth=2.5,
                   label=f'Mean PR (CV AP = {mean_ap:.3f} ± {std_ap:.3f})',
                   zorder=3)
            
            # Plot standard deviation band
            ax.fill_between(mean_recall, precision_lower, precision_upper, color='darkblue', alpha=0.2,
                           label=f'±1 std. dev.')
            
            # Plot baseline (random classifier)
            ax.axhline(y=baseline, linestyle='--', color='gray', alpha=0.8,
                      label=f'Baseline (AP = {baseline:.3f})')
            
            # Configure plot
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
            ax.set_ylabel('Precision (PPV)', fontsize=12)
            ax.set_title(f'Cross-Validated Precision-Recall Curve - {model.__class__.__name__}\n'
                        f'({n_splits}-fold × {n_repeats} repeats)', fontsize=14)
            ax.legend(loc='lower left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

        # Plot Calibration Curve if requested
        if plot_calibration_curve:
            from scipy import interpolate
            
            # Lists to store calibration data from each fold
            prob_true_interp_list = []
            
            # Create common probability bins for interpolation
            common_prob_bins = np.linspace(0, 1, 11)  # 10 bins = 11 points
            
            # Perform cross-validation to collect calibration curves from each fold
            for train_index, val_index in cv.split(X_train, y_train):
                model_fold = clone(model)
                model_fold.fit(X_train[train_index], y_train[train_index])
                
                y_proba = model_fold.predict_proba(X_train[val_index])[:, 1]
                y_true = y_train[val_index]
                
                # Calculate calibration curve for this fold
                prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
                
                # Interpolate to common probability bins
                # Only interpolate if we have at least 2 points
                if len(prob_pred) >= 2:
                    # Use linear interpolation
                    interp_func = interpolate.interp1d(
                        prob_pred, prob_true, 
                        kind='linear', 
                        bounds_error=False, 
                        fill_value=(prob_true[0], prob_true[-1])
                    )
                    prob_true_interp = interp_func(common_prob_bins)
                    prob_true_interp_list.append(prob_true_interp)
            
            # Check if we have any valid calibration data
            if len(prob_true_interp_list) > 0:
                # Convert to array for calculations
                prob_true_array = np.array(prob_true_interp_list)
                
                # Calculate mean and std
                mean_prob_true = np.mean(prob_true_array, axis=0)
                std_prob_true = np.std(prob_true_array, axis=0)
                
                # Plot Calibration curve
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot perfectly calibrated line
                ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
                
                # Plot mean calibration curve with error bars
                ax.errorbar(common_prob_bins, mean_prob_true, yerr=std_prob_true,
                           fmt='o-', color='darkgreen', linewidth=2, markersize=8,
                           capsize=5, label='Mean Calibration (±1 std)',
                           ecolor='lightgreen', alpha=0.8)
                
                # Configure plot
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.0])
                ax.set_xlabel('Mean Predicted Probability', fontsize=12)
                ax.set_ylabel('Fraction of Positives (Actual)', fontsize=12)
                ax.set_title(f'Cross-Validated Calibration Curve (Reliability Diagram) - {model.__class__.__name__}\n'
                            f'({n_splits}-fold × {n_repeats} repeats)', fontsize=14)
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()

        # Plot Decision Threshold Analysis if requested
        if plot_threshold_analysis:
            # Define threshold range
            thresholds = np.linspace(0, 1, 101)
            
            # Lists to store metrics at each threshold
            precision_cv = []
            recall_cv = []
            f1_cv = []
            
            # Perform cross-validation to collect metrics at different thresholds
            for threshold in thresholds:
                precision_fold = []
                recall_fold = []
                f1_fold = []
                
                for train_index, val_index in cv.split(X_train, y_train):
                    model_fold = clone(model)
                    model_fold.fit(X_train[train_index], y_train[train_index])
                    
                    y_proba = model_fold.predict_proba(X_train[val_index])[:, 1]
                    y_true = y_train[val_index]
                    
                    # Apply threshold
                    y_pred_thresh = (y_proba >= threshold).astype(int)
                    
                    # Calculate metrics (handle division by zero)
                    if np.sum(y_pred_thresh) == 0:
                        precision = 0
                    else:
                        precision = precision_score(y_true, y_pred_thresh, zero_division=0)
                    
                    recall = recall_score(y_true, y_pred_thresh, zero_division=0)
                    
                    if precision + recall == 0:
                        f1 = 0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    
                    precision_fold.append(precision)
                    recall_fold.append(recall)
                    f1_fold.append(f1)
                
                precision_cv.append(np.mean(precision_fold))
                recall_cv.append(np.mean(recall_fold))
                f1_cv.append(np.mean(f1_fold))
            
            # Convert to arrays
            precision_cv = np.array(precision_cv)
            recall_cv = np.array(recall_cv)
            f1_cv = np.array(f1_cv)
            
            # Find optimal thresholds
            best_f1_idx = np.argmax(f1_cv)
            best_f1_threshold = thresholds[best_f1_idx]
            best_f1_score = f1_cv[best_f1_idx]
            
            # Plot Threshold Analysis
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot metrics vs threshold
            ax.plot(thresholds, precision_cv, 'b-', linewidth=2.5, label='Precision', alpha=0.8)
            ax.plot(thresholds, recall_cv, 'g-', linewidth=2.5, label='Recall', alpha=0.8)
            ax.plot(thresholds, f1_cv, 'r-', linewidth=2.5, label='F1-Score', alpha=0.8)
            
            # Mark optimal F1 threshold
            ax.axvline(x=best_f1_threshold, color='purple', linestyle='--', linewidth=2,
                      label=f'Optimal F1 = {best_f1_score:.3f} @ threshold = {best_f1_threshold:.2f}')
            ax.scatter([best_f1_threshold], [best_f1_score], color='purple', s=150, zorder=5)
            
            # Configure plot
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Classification Threshold', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'Decision Threshold Analysis - {model.__class__.__name__}\n'
                        f'({n_splits}-fold × {n_repeats} repeats)', fontsize=14)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

        return metrics

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

def compare_cv_metrics(metrics_dict: dict) -> pd.DataFrame:
    """
    Compare cross-validated metrics from multiple classification models in a single table.
    
    This function extracts only the cross-validation metrics (not single-run metrics)
    from multiple model evaluations and presents them in a comparison DataFrame.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with model names as keys and metrics dictionaries as values.
        Format: {'model_name': metrics_dict_from_eval_classification}
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with models as rows and CV metrics as columns
        
    Example:
    --------
    >>> metrics_rf = eval_classification(model=rf, X_train=X_train, y_train=y_train, 
    ...                                  X_test=X_test, y_test=y_test)
    >>> metrics_xgb = eval_classification(model=xgb, X_train=X_train, y_train=y_train,
    ...                                   X_test=X_test, y_test=y_test)
    >>> comparison = compare_cv_metrics({
    ...     'Random Forest': metrics_rf,
    ...     'XGBoost': metrics_xgb
    ... })
    >>> display(comparison)
    """
    cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    data = []
    
    for model_name, metrics in metrics_dict.items():
        row = {'Model': model_name}
        
        for metric in cv_metrics:
            if 'cv' in metrics and metric in metrics['cv']:
                mean_val = metrics['cv'][metric]['test_mean']
                std_val = metrics['cv'][metric]['test_std']
                row[f'{metric.upper()}'] = f"{mean_val*100:.2f}% ± {std_val*100:.2f}%"
            else:
                row[f'{metric.upper()}'] = 'N/A'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    return df

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║            Functions for Hyperparameter Tuning                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

# Function for tuning multiple pipelines
def tune_pipelines(
    pipedict: dict,
    param_grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    scoring='accuracy',
    search_method='grid',
    n_iter=50,
    n_splits=5,
    n_repeats=3,
    random_state=42,
    display=True,
    n_jobs=-1,
    plot_validation_curves=False,
    plot_learning_curves=True,
    progress_bar=False
) -> tuple:
    """
    Perform hyperparameter tuning with grid/random search and optional validation curve plots.

    Parameters:
    -----------
    pipedict : dict
        Dictionary of pipeline names and their corresponding pipeline objects
    param_grid : dict
        Dictionary of parameter grids for each pipeline
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    scoring : str, optional (default='accuracy')
        Metric to optimize during search
    search_method : str, optional (default='grid')
        Search method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    n_iter : int, optional (default=50)
        Number of parameter combinations to try (only for random search)
    n_splits : int, optional (default=5)
        Number of splits for cross-validation
    n_repeats : int, optional (default=3)
        Number of times to repeat cross-validation
    random_state : int, optional (default=42)
        Seed for reproducibility
    display : bool, optional (default=True)
        Whether to display detailed results
    n_jobs : int, optional (default=-1)
        Number of CPU cores to use
    plot_validation_curves : bool, optional (default=False)
        Whether to plot validation curves (score vs hyperparameter value) for each parameter.
        Note: This fits models separately from the main search and can be slow.
    plot_learning_curves : bool, optional (default=True)
        Whether to plot learning curves (score vs training set size) after hyperparameter tuning.
        Uses the best estimator from the search to diagnose underfitting/overfitting and 
        determine if more data would help. Shows training score vs validation score across
        different training set sizes.
    progress_bar : bool, optional (default=False)
        Whether to display a progress bar for parameter evaluations

    Returns:
    --------
    tuple
        (dictionary of fitted search objects, list of fitting times)
    
    Example:
    --------
    >>> fitted_models, times = tune_pipelines(
    >>>     pipedict={'rf': RandomForestClassifier()},
    >>>     param_grid={'rf': {'n_estimators': [10, 50, 100]}},
    >>>     X_train=X_train, y_train=y_train,
    >>>     X_test=X_test, y_test=y_test,
    >>>     search_method='random', n_iter=20
    >>> )
    """
    fitted_models = {}
    fit_times = []

    # Wrap pipeline iteration with optional tqdm
    pipe_iter = tqdm(pipedict.items(), desc="Processing Pipelines",
                    position=0, leave=True) if progress_bar else pipedict.items()

    for name, pipeline in pipe_iter:
        try:
            # Plot validation curves if requested (shows score vs hyperparameter value)
            if plot_validation_curves:
                _plot_validation_curves(
                    pipeline=pipeline,
                    param_grid=param_grid[name],
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    scoring=scoring,
                    name=name,
                    display=display,
                    progress_bar=progress_bar
                )

            # Construct cross-validation strategy
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state
            )

            # Choose search method
            if search_method == 'random':
                model = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid[name],
                    n_iter=n_iter,
                    scoring=scoring,
                    cv=cv,
                    verbose=2 if display else 0,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    return_train_score=True,
                    error_score='raise'
                )
            else:  # grid search
                model = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid[name],
                    scoring=scoring,
                    cv=cv,
                    verbose=2 if display else 0,
                    n_jobs=n_jobs,
                    return_train_score=True,
                    error_score='raise'
                )

            # Fit with timing
            start_time = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start_time
            fit_times.append(round(fit_time, 2))

            # Store fitted model
            fitted_models[name] = model

            if display:
                print(f"\n{'='*50}")
                print(f"Results for {name} ({search_method.upper()} Search)")
                print(f"{'='*50}")
                print(f"Fit Time: {fit_time:.2f}s")
                print(f"Best CV {scoring}: {model.best_score_:.4f}")
                print("\nBest Parameters:")
                for param, value in model.best_params_.items():
                    print(f"  {param}: {value}")
                
                # Show test score with best estimator
                if scoring == 'roc_auc':
                    test_score = roc_auc_score(y_test, model.best_estimator_.predict_proba(X_test)[:, 1])
                else:
                    test_score = get_scorer(scoring)._score_func(y_test, model.best_estimator_.predict(X_test))
                print(f"\nTest {scoring}: {test_score:.4f}")
                
                # Plot learning curves if requested
                if plot_learning_curves:
                    print(f"\nGenerating learning curves for {name}...")
                    _plot_learning_curves(
                        estimator=model.best_estimator_,
                        X_train=X_train,
                        y_train=y_train,
                        scoring=scoring,
                        name=name,
                        n_splits=n_splits,
                        n_repeats=n_repeats,
                        random_state=random_state
                    )

        except Exception as e:
            print(f"Error during {name} pipeline tuning: {str(e)}")
            raise

    return fitted_models, fit_times

# Function for plotting validation curves
def _plot_validation_curves(
    pipeline,
    param_grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    scoring: str,
    name: str,
    display: bool = True,
    progress_bar: bool = False
):
    """
    Plot validation curves showing score vs hyperparameter value for each parameter.
    
    This is an internal helper function used by tune_pipelines.
    """
    # Wrap param_grid iteration with optional tqdm
    param_iter = tqdm(param_grid.items(), desc=f"Validation Curves for {name}",
                    position=1, leave=True) if progress_bar else param_grid.items()

    for param_name, param_values in param_iter:
        if not isinstance(param_values, (list, np.ndarray)):
            continue

        # Skip parameters with too few values
        if len(param_values) < 2:
            continue

        # Initialize score lists
        train_scores = []
        test_scores = []

        if display:
            print(f"\nValidation curve for {param_name}...")

        for value in param_values:
            # Create a copy of the pipeline with the current parameter value
            current_pipeline = clone(pipeline)
            current_pipeline.set_params(**{param_name: value})

            # Fit and evaluate
            current_pipeline.fit(X_train, y_train)

            # Get predictions and scores
            if scoring == 'roc_auc':
                y_pred_train = current_pipeline.predict_proba(X_train)[:, 1]
                y_pred_test = current_pipeline.predict_proba(X_test)[:, 1]
                score_func = roc_auc_score
            else:
                y_pred_train = current_pipeline.predict(X_train)
                y_pred_test = current_pipeline.predict(X_test)
                score_func = get_scorer(scoring)._score_func

            train_score = score_func(y_train, y_pred_train)
            test_score = score_func(y_test, y_pred_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

            if display:
                print(f'  {param_name}={value}: train={train_score:.3f}, test={test_score:.3f}')

        # Plot validation curve
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, train_scores, 'o-', label='Train', linewidth=2)
        plt.plot(param_values, test_scores, 'o-', label='Test', linewidth=2)

        # Find best parameter value
        best_idx = np.argmax(test_scores)
        best_value = param_values[best_idx]
        best_score = test_scores[best_idx]

        # Add vertical line and annotation for best value
        plt.axvline(x=best_value, color='r', linestyle='--', alpha=0.5)
        plt.annotate(
            f'Best: {best_value}\nScore: {best_score:.3f}',
            xy=(best_value, best_score),
            xytext=(10, -30), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        plt.title(f'Validation Curve: {name} | {param_name}')
        plt.xlabel(param_name)
        plt.ylabel(scoring.upper())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Function for plotting learning curves
def _plot_learning_curves(
    estimator,
    X_train,
    y_train,
    scoring: str,
    name: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42
):
    """
    Plot learning curves showing training and validation scores vs training set size.
    
    This helps diagnose underfitting/overfitting and determine if more data would help.
    Uses RepeatedStratifiedKFold for robust cross-validation.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        The fitted best estimator from hyperparameter tuning
    X_train, y_train : array-like
        Training data
    scoring : str
        Scoring metric to use
    name : str
        Model name for the plot title
    n_splits : int, optional (default=5)
        Number of CV splits
    n_repeats : int, optional (default=3)
        Number of CV repeats
    random_state : int, optional (default=42)
        Random seed for reproducibility
    """
    # Create CV strategy
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    
    # Define training set sizes (percentages of training data)
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curve
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        return_times=False
    )
    
    # Calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot training scores
    ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue', 
            label='Training Score', linewidth=2.5, markersize=8)
    ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color='blue')
    
    # Plot validation scores
    ax.plot(train_sizes_abs, test_scores_mean, 'o-', color='darkorange',
            label='Validation Score', linewidth=2.5, markersize=8)
    ax.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2, color='darkorange')
    
    # Add annotations for gap analysis
    final_train = train_scores_mean[-1]
    final_val = test_scores_mean[-1]
    gap = final_train - final_val
    
    # Add text box with gap analysis
    textstr = f'Final Gap: {gap:.3f}\n'
    if gap < 0.01:
        textstr += 'Status: Well-fitted ✓'
    elif gap < 0.05:
        textstr += 'Status: Slight overfitting'
    else:
        textstr += 'Status: High variance (overfitting)'
    
    if test_scores_mean[-1] - test_scores_mean[0] > 0.05:
        textstr += '\nMore data may help'
    
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Configure plot
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel(f'{scoring.upper()} Score', fontsize=12)
    ax.set_title(f'Learning Curves - {name}\n(Best Model After Hyperparameter Tuning)', 
                fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Define supported models
SUPPORTED_MODELS = ['logisticregression', 'knn', 'decisiontree', 'randomforest', 'gb', 'xgboost']

# Function for getting model pipeline
def get_model_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """
    Create a pipeline for a specific model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model. Supported: 'logisticregression', 'knn', 'decisiontree',
        'randomforest', 'gb', 'xgboost'
    random_state : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    Pipeline
        Scikit-learn pipeline with the specified model
        
    Raises:
    -------
    ValueError
        If model_name is not supported
    """
    pipelines = {
        'logisticregression': Pipeline([
            ('lr', LogisticRegression(random_state=random_state))
        ]),
        'knn': Pipeline([
            ('knn', KNeighborsClassifier())
        ]),
        'decisiontree': Pipeline([
            ('dt', DecisionTreeClassifier(random_state=random_state))
        ]),
        'randomforest': Pipeline([
            ('rf', RandomForestClassifier(random_state=random_state))
        ]),
        'gb': Pipeline([
            ('gb', GradientBoostingClassifier(random_state=random_state))
        ]),
        'xgboost': Pipeline([
            ('xgb', XGBClassifier(random_state=random_state))
        ])
    }
    
    model_key = model_name.lower()
    if model_key not in pipelines:
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {SUPPORTED_MODELS}")
    
    return pipelines[model_key]

# Function for getting hyperparameters
def get_hyperparameters(model_name: str, search_method: str = 'grid') -> dict:
    """
    Get hyperparameter grid for a specific model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to get hyperparameters for
    search_method : str, optional (default='grid')
        'grid' returns smaller grids suitable for exhaustive search,
        'random' returns larger distributions for random sampling
        
    Returns:
    --------
    dict
        Dictionary of hyperparameters for search
        
    Raises:
    -------
    ValueError
        If model_name is not supported
    """
    # Smaller grids for GridSearchCV (exhaustive)
    grid_params = {
        'logisticregression': {
            'lr__C': [0.001, 0.01, 0.1, 0.5, 1.0],
            'lr__penalty': ['l2'],
            'lr__solver': ['lbfgs', 'liblinear'],
            'lr__max_iter': [1000, 5000]
        },
        'knn': {
            'knn__n_neighbors': [3, 5, 7, 11, 15, 21],
            'knn__weights': ['uniform', 'distance'],
            'knn__p': [1, 2],
            'knn__algorithm': ['auto'],
            'knn__leaf_size': [20, 30]
        },
        'decisiontree': {
            'dt__criterion': ['entropy', 'gini'],
            'dt__max_depth': [3, 5, 7, 10, 15, 20],
            'dt__min_samples_split': [2, 5, 10, 20],
            'dt__min_samples_leaf': [1, 2, 5, 10],
            'dt__max_features': ['sqrt']
        },
        'randomforest': {
            'rf__n_estimators': [50, 100, 200],
            'rf__criterion': ['entropy', 'gini'],
            'rf__max_depth': [5, 10, 15, 20],
            'rf__min_samples_split': [2, 5, 10],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__min_samples_leaf': [1, 2, 4]
        },
        'gb': {
            'gb__n_estimators': [50, 100, 200],
            'gb__max_depth': [3, 5, 7],
            'gb__min_samples_split': [2, 5],
            'gb__min_samples_leaf': [1, 2],
            'gb__learning_rate': [0.01, 0.1, 0.2]
        },
        'xgboost': {
            'xgb__learning_rate': [0.01, 0.1, 0.2],
            'xgb__n_estimators': [50, 100, 200],
            'xgb__max_depth': [3, 5, 7, 10],
            'xgb__tree_method': ['hist']
        }
    }
    
    # Larger distributions for RandomizedSearchCV
    random_params = {
        'logisticregression': {
            'lr__C': [float(x) for x in np.logspace(-3, 1, 50)],
            'lr__penalty': ['l2'],
            'lr__solver': ['newton-cg', 'lbfgs', 'newton-cholesky', 'liblinear'],
            'lr__max_iter': [1000, 5000, 10000]
        },
        'knn': {
            'knn__n_neighbors': list(range(1, 31)),
            'knn__weights': ['uniform', 'distance'],
            'knn__p': [1, 2],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'knn__leaf_size': list(range(10, 50, 5))
        },
        'decisiontree': {
            'dt__criterion': ['entropy', 'gini'],
            'dt__max_depth': list(range(1, 25)),
            'dt__min_samples_split': list(range(2, 50)),
            'dt__min_samples_leaf': list(range(1, 30)),
            'dt__max_features': ['sqrt', 'log2', None]
        },
        'randomforest': {
            'rf__n_estimators': [25, 50, 75, 100, 150, 200, 300],
            'rf__criterion': ['entropy', 'gini'],
            'rf__max_depth': list(range(3, 25)),
            'rf__min_samples_split': [2, 5, 7, 10, 15],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__min_samples_leaf': [1, 2, 4, 6, 8]
        },
        'gb': {
            'gb__n_estimators': [25, 50, 100, 150, 200, 300],
            'gb__max_depth': list(range(1, 15)),
            'gb__min_samples_split': [2, 3, 5, 7, 10],
            'gb__min_samples_leaf': [1, 2, 3, 5, 7],
            'gb__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
        },
        'xgboost': {
            'xgb__learning_rate': [float(x) for x in np.linspace(0.01, 0.3, 20)],
            'xgb__n_estimators': [25, 50, 75, 100, 150, 200],
            'xgb__max_depth': list(range(3, 15)),
            'xgb__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'xgb__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'xgb__tree_method': ['hist']
        }
    }
    
    model_key = model_name.lower()
    params = random_params if search_method == 'random' else grid_params
    
    if model_key not in params:
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {SUPPORTED_MODELS}")
    
    return params[model_key]

# Function for tuning a single model
def tune_single_model(
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    scoring='accuracy',
    search_method='grid',
    n_iter=50,
    n_splits=5,
    n_repeats=3,
    random_state=42,
    display=True,
    plot_validation_curves=False,
    plot_learning_curves=True,
    progress_bar=False
) -> tuple:
    """
    Tune a single model using grid or random search.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to tune. Supported: 'logisticregression', 'knn', 
        'decisiontree', 'randomforest', 'gb', 'xgboost'
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    scoring : str, optional (default='accuracy')
        Metric to optimize
    search_method : str, optional (default='grid')
        'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    n_iter : int, optional (default=50)
        Number of iterations for random search
    n_splits : int, optional (default=5)
        Number of CV splits
    n_repeats : int, optional (default=3)
        Number of CV repeats
    random_state : int, optional (default=42)
        Random seed
    display : bool, optional (default=True)
        Whether to print results
    plot_validation_curves : bool, optional (default=False)
        Whether to plot validation curves
    plot_learning_curves : bool, optional (default=True)
        Whether to plot learning curves after tuning
    progress_bar : bool, optional (default=False)
        Whether to show progress bar
        
    Returns:
    --------
    tuple
        (fitted search object, fitting time in seconds)
        
    Example:
    --------
    >>> model, time = tune_single_model(
    >>>     'randomforest', X_train, y_train, X_test, y_test,
    >>>     search_method='random', n_iter=30
    >>> )
    >>> print(model.best_params_)
    """
    # Get pipeline and hyperparameters for the specified model
    pipeline = get_model_pipeline(model_name, random_state=random_state)
    params = get_hyperparameters(model_name, search_method=search_method)
    
    # Create single-model dictionary
    pipedict = {model_name: pipeline}
    param_grid = {model_name: params}
    
    # Run search
    fitted_models, fit_times = tune_pipelines(
        pipedict=pipedict,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scoring=scoring,
        search_method=search_method,
        n_iter=n_iter,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        display=display,
        plot_validation_curves=plot_validation_curves,
        plot_learning_curves=plot_learning_curves,
        progress_bar=progress_bar
    )
    
    return fitted_models[model_name], fit_times[0]

# Function for tuning all models
def tune_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models: list = None,
    scoring='accuracy',
    search_method='grid',
    n_iter=50,
    n_splits=5,
    n_repeats=3,
    random_state=42,
    display=True,
    plot_learning_curves=True,
    progress_bar=True
) -> tuple:
    """
    Tune multiple classification models using grid or random search.
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    models : list, optional (default=None)
        List of model names to tune. If None, tunes all supported models:
        ['logisticregression', 'knn', 'decisiontree', 'randomforest', 'gb', 'xgboost']
    scoring : str, optional (default='accuracy')
        Metric to optimize
    search_method : str, optional (default='grid')
        'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    n_iter : int, optional (default=50)
        Number of iterations for random search
    n_splits : int, optional (default=5)
        Number of CV splits
    n_repeats : int, optional (default=3)
        Number of CV repeats
    random_state : int, optional (default=42)
        Random seed
    display : bool, optional (default=True)
        Whether to print results
    plot_learning_curves : bool, optional (default=True)
        Whether to plot learning curves for each model after tuning
    progress_bar : bool, optional (default=True)
        Whether to show progress bar
        
    Returns:
    --------
    tuple
        (dictionary of fitted search objects, list of fitting times)
        
    Example:
    --------
    >>> # Tune only specific models
    >>> fitted, times = tune_all_models(
    >>>     X_train, y_train, X_test, y_test,
    >>>     models=['randomforest', 'xgboost'],
    >>>     search_method='random', n_iter=30
    >>> )
    """
    if models is None:
        models = SUPPORTED_MODELS

    # Build pipelines and params for selected models
    all_pipelines = {
        name: get_model_pipeline(name, random_state=random_state)
        for name in models
    }
    all_hyperparameters = {
        name: get_hyperparameters(name, search_method=search_method)
        for name in models
    }

    # Run search
    fitted_models, times = tune_pipelines(
        pipedict=all_pipelines,
        param_grid=all_hyperparameters,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scoring=scoring,
        search_method=search_method,
        n_iter=n_iter,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        display=display,
        plot_learning_curves=plot_learning_curves,
        progress_bar=progress_bar
    )

    return fitted_models, times