"""GPU feature package (RTX 4070 Laptop POC; edge path uses cpu_* reference)."""
from .features import FEATURE_NAMES, cpu_window_features

__all__ = ["FEATURE_NAMES", "cpu_window_features"]
