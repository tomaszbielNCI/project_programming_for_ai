"""
Bayesian Neural Network models for financial time series prediction.
"""

# Import models to make them available at the package level
from .bnn.model import build_bnn_showcase

__all__ = ['build_bnn_showcase']
