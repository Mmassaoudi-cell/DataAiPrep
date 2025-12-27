#!/usr/bin/env python3
"""
DataAiPrep - Advanced ML Data Quality Assessment Platform
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Core dependencies
install_requires = [
    "PyQt6>=6.6.0",
    "PyQt6-SVG>=6.6.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "joblib>=1.3.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.17.0",
    "scipy>=1.11.0",
    "statsmodels>=0.14.0",
    "tqdm>=4.66.0",
    "python-dateutil>=2.8.0",
    "colorama>=0.4.0",
    "psutil>=5.9.0",
]

# Optional dependencies
extras_require = {
    "web": [
        "fastapi>=0.110.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.5.0",
        "python-multipart>=0.0.6",
    ],
    "data": [
        "openpyxl>=3.1.0",
        "xlrd>=2.0.0",
        "pyarrow>=14.0.0",
        "fastparquet>=0.8.0",
        "sqlalchemy>=2.0.0",
    ],
    "reporting": [
        "jinja2>=3.1.0",
        "weasyprint>=60.0",
        "reportlab>=4.0.0",
    ],
    "shap": [
        "shap>=0.44.0",
    ],
    "dask": [
        "dask[complete]>=2023.12.0",
    ],
    "mlops": [
        "mlflow>=2.9.0",
        "wandb>=0.16.0",
    ],
    "full": [
        # Web
        "fastapi>=0.110.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.5.0",
        "python-multipart>=0.0.6",
        # Data formats
        "openpyxl>=3.1.0",
        "xlrd>=2.0.0",
        "pyarrow>=14.0.0",
        "fastparquet>=0.8.0",
        "sqlalchemy>=2.0.0",
        # Reporting
        "jinja2>=3.1.0",
        "weasyprint>=60.0",
        "reportlab>=4.0.0",
        # Advanced features
        "shap>=0.44.0",
        "dask[complete]>=2023.12.0",
        "mlflow>=2.9.0",
        "wandb>=0.16.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.5.0",
        "pre-commit>=3.4.0",
    ],
}

setup(
    name="dataiprep",
    version="2.0.0",
    author="Mohamed Massaoudi",
    author_email="mohamed.massaoudi@tamu.edu",
    description="Advanced ML Data Quality Assessment Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/massaoudi-lab/dataiprep",
    project_urls={
        "Bug Tracker": "https://github.com/massaoudi-lab/dataiprep/issues",
        "Documentation": "https://github.com/massaoudi-lab/dataiprep#readme",
        "Source Code": "https://github.com/massaoudi-lab/dataiprep",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Quality Assurance",
    ],
    keywords=[
        "data-quality",
        "machine-learning",
        "data-preprocessing",
        "feature-selection",
        "shap",
        "explainability",
        "data-leakage",
        "mlops",
        "data-science",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    package_data={
        "": ["*.svg", "*.png", "*.html", "*.css", "*.js"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "dataiprep=main:main",
        ],
    },
    zip_safe=False,
)

