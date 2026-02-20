# Import metrics score that will be evaluated
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    get_scorer,
    confusion_matrix
)
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
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
        progress_bar=progress_bar
    )

    return fitted_models, times