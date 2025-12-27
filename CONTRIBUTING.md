# Contributing to DataAiPrep

Thank you for your interest in contributing to DataAiPrep! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating a new issue
3. **Include**:
   - Python version and OS
   - DataAiPrep version
   - Minimal reproducible example
   - Expected vs actual behavior
   - Full error traceback

### Suggesting Features

1. **Check existing issues** and discussions
2. **Use the feature request template**
3. **Describe**:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered
   - Potential impact on existing functionality

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding standards** (see below)
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** before submitting
6. **Write clear commit messages**

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/dataiprep.git
cd dataiprep

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_feature_selection.py

# Run specific test
pytest tests/test_feature_selection.py::test_boruta_selection
```

### Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src tests
isort src tests

# Check linting
flake8 src tests

# Type checking
mypy src
```

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for all public functions and classes

### Docstring Format

Use Google-style docstrings:

```python
def analyze_data(df: pd.DataFrame, target: str = None) -> dict:
    """Analyze data quality and return results.
    
    Args:
        df: Input DataFrame to analyze.
        target: Optional target column name for supervised analysis.
    
    Returns:
        Dictionary containing analysis results with keys:
            - 'summary': Overall quality summary
            - 'issues': List of detected issues
            - 'recommendations': Suggested actions
    
    Raises:
        ValueError: If DataFrame is empty.
        KeyError: If target column doesn't exist.
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
        >>> results = analyze_data(df)
        >>> print(results['summary'])
    """
```

### Commit Messages

Follow the conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(feature-selection): add mRMR algorithm

fix(gui): resolve threading issue in analysis worker

docs(readme): add installation instructions for conda
```

## Project Structure

```
dataiprep/
├── main.py                 # Entry point
├── src/
│   ├── advanced/           # Advanced analysis modules
│   │   ├── feature_selection.py
│   │   ├── shap_explainer.py
│   │   ├── advanced_leakage.py
│   │   └── ...
│   ├── analysis/           # Core analysis modules
│   ├── gui/                # PyQt6 GUI components
│   ├── pipeline/           # Preprocessing pipeline
│   └── ...
├── tests/                  # Test files
│   ├── test_feature_selection.py
│   └── ...
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## Adding New Features

### Adding a New Analysis Module

1. Create the module in `src/advanced/` or `src/analysis/`
2. Add exports to `__init__.py`
3. Add CLI support in `main.py` if applicable
4. Write unit tests in `tests/`
5. Update documentation

Example module structure:

```python
"""
Module description.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class NewAnalyzer:
    """Brief description of the analyzer.
    
    Attributes:
        param1: Description of param1.
        param2: Description of param2.
    """
    
    def __init__(self, param1: float = 0.5, param2: str = "default"):
        """Initialize the analyzer.
        
        Args:
            param1: Description with default value note.
            param2: Description with default value note.
        """
        self.param1 = param1
        self.param2 = param2
        self.results_ = None
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """Perform analysis on the data.
        
        Args:
            data: Input DataFrame.
        
        Returns:
            Dictionary with analysis results.
        """
        # Implementation
        self.results_ = {}
        return self.results_
    
    def generate_report(self) -> str:
        """Generate a text report of the analysis.
        
        Returns:
            Formatted report string.
        """
        if self.results_ is None:
            return "No analysis has been run yet."
        
        # Generate report
        return "Report content"
```

### Adding New CLI Commands

1. Add argument parser in `main.py`
2. Create the command function
3. Add to the command dispatch logic
4. Update help text and examples

## Testing Guidelines

### Test Structure

```python
import pytest
import pandas as pd
from src.advanced import NewAnalyzer


class TestNewAnalyzer:
    """Tests for NewAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_initialization(self):
        """Test default initialization."""
        analyzer = NewAnalyzer()
        assert analyzer.param1 == 0.5
        assert analyzer.param2 == "default"
    
    def test_analyze_basic(self, sample_data):
        """Test basic analysis functionality."""
        analyzer = NewAnalyzer()
        results = analyzer.analyze(sample_data)
        
        assert isinstance(results, dict)
        assert 'summary' in results
    
    def test_analyze_empty_data(self):
        """Test handling of empty DataFrame."""
        analyzer = NewAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.analyze(pd.DataFrame())
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update CHANGELOG.md for releases
- Create examples for new features

## Release Process

1. Update version in `setup.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create a pull request to `main`
4. After merge, create a GitHub release
5. Package will be automatically published to PyPI

## Questions?

- Open a [GitHub Discussion](https://github.com/massaoudi-lab/dataiprep/discussions)
- Email: mohamed.massaoudi@tamu.edu

Thank you for contributing to DataAiPrep! 🎉

