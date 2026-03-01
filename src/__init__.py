# src/__init__.py
"""
Moodle Analytics Pipeline
-------------------------
This package contains modules for:
- pipeline: sessionization + feature synthesis
- eda_plots: visualization functions
- regression: regression models
- importance: feature importance analysis
"""

from .pipeline import AcademicMoodlePipeline
from .eda_plots import run_eda_plots
from .regression import run_regression_models
from .importance import feature_importance_analysis