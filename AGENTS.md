# AGENTS.md - Agentic Coding Guidelines

This file provides guidelines for AI agents working in this repository.

## Project Overview

This is a Python data science utility library providing functions for EDA, preprocessing, statistics, visualization, and ML evaluation. It uses `uv` for package management and requires Python >=3.12.

## Build/Lint/Test Commands

### Running Tests
```bash
# Run all tests
uv run python -m pytest tests/

# Run a single test file
uv run python tests/test_eval_classification.py

# Run a specific test function
uv run python -c "from tests.test_ab_testing_manual import test_sample_size; test_sample_size()"

# Run with pytest (if installed)
uv run pytest tests/test_eval_classification.py -v
```

### Package Management
```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Update lock file
uv lock
```

### Running Python Files
```bash
# Run any Python file
uv run python <filename>.py

# Run with Jupyter
uv run jupyter notebook
```

## Code Style Guidelines

### Imports
- **Standard library** imports first (e.g., `sys`, `os`, `time`)
- **Third-party** imports second (e.g., `numpy`, `pandas`, `sklearn`)
- **Local** imports last (e.g., `from utils import ...`)
- Group imports with a blank line between each group
- Use `from typing import ...` for type hints
- Always use absolute imports with explicit paths

### Formatting
- Use **4 spaces** for indentation
- **Line length**: Keep lines under 100 characters when possible
- Use **double quotes** for strings consistently
- Add **two blank lines** before top-level function definitions
- Add **one blank line** between methods in a class
- Use **trailing commas** in multi-line collections

### Type Hints
- Use type hints for all function parameters and return values
- Import types from `typing` module: `List`, `Dict`, `Union`, `Optional`, `Any`, `Tuple`
- Example: `def func(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:`
- Use `Optional[Type]` for parameters that can be None

### Naming Conventions
- **Functions**: `snake_case` (e.g., `check_data_information`)
- **Variables**: `snake_case` (e.g., `n_samples`, `X_train`)
- **Constants**: `UPPER_CASE` (e.g., `RANDOM_STATE`)
- **Classes**: `PascalCase` (if any)
- **Private functions**: `_leading_underscore` (if any)
- **Modules**: `snake_case.py`

### Docstrings
Use Google-style docstrings with the following structure:
```python
def function_name(param: Type) -> ReturnType:
    """
    Brief description of what the function does.
    
    Parameters:
    -----------
    param : Type
        Description of the parameter
    
    Returns:
    --------
    ReturnType
        Description of the return value
    
    Examples:
    ---------
    >>> # Example usage
    >>> result = function_name(value)
    """
```

### Section Headers
Use decorative box-style comments for module sections:
```python
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Section Name Description                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
```

### Error Handling
- Use `try/except` blocks for operations that may fail
- Catch specific exceptions, not bare `except:`
- Provide meaningful error messages
- Use warnings for non-critical issues: `import warnings; warnings.warn("message")`

### Function Design
- Keep functions focused on a single responsibility
- Use descriptive parameter names (e.g., `X_train`, `y_test` for ML data)
- Provide sensible defaults for optional parameters
- Use `**kwargs` for passing additional parameters to underlying functions

### Random State
- Always accept a `random_state` parameter (default: 42) in stochastic functions
- Pass `random_state` to all sklearn functions that accept it

### Comments
- Use inline comments sparingly, prefer self-documenting code
- Add comments to explain "why" not "what"
- Keep comments up-to-date with code changes

### File Structure
- Place reusable utilities in `utils/` directory
- Place tests in `tests/` directory
- Place notebooks in `notebooks/` directory
- Place data files in `data/` directory

## Testing Guidelines

### Test File Structure
```python
import sys
import os

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import statements
import pandas as pd
import numpy as np
from utils.module import function_name

# Test code
```

### Writing Tests
- Create standalone test files that can be run directly
- Use synthetic data (e.g., `make_classification`, `make_regression`)
- Set `random_state` for reproducibility
- Print meaningful output to verify functionality
- Test both success and edge cases

## Dependencies

Core dependencies (see `pyproject.toml`):
- numpy, pandas, scipy
- scikit-learn, xgboost
- matplotlib, seaborn
- statsmodels
- tqdm, ipykernel

Always use `uv` commands for dependency management instead of `pip`.

## Jupyter Notebooks

- Place notebooks in the `notebooks/` directory
- Use meaningful names (e.g., `classification.ipynb`)
- Import utilities from `utils` module
- Keep cells runnable from top to bottom
