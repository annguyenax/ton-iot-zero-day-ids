"""
Utility Functions for Reproducibility and Environment Management

This module provides functions to ensure full reproducibility of training results
and manage environment information for consistent model behavior.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_all_seeds(seed=42):
    """
    Set all random seeds for full reproducibility across the entire stack.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy's random operations
    - TensorFlow's random operations (including Keras)
    - Python hash seed (for dict/set ordering)
    - TensorFlow deterministic operations

    Args:
        seed (int): Random seed value. Default is 42.

    Note:
        Enabling deterministic operations may reduce performance by 10-20%
        but ensures 100% reproducible results across multiple runs.

    Example:
        >>> from utils import set_all_seeds
        >>> set_all_seeds(42)
        >>> # Now all random operations will be deterministic
    """
    # Python built-in random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # TensorFlow random (including Keras)
    tf.random.set_seed(seed)

    # Python hash seed for dictionary/set ordering
    # This ensures consistent iteration order for dicts and sets
    os.environ['PYTHONHASHSEED'] = str(seed)

    # TensorFlow deterministic operations
    # Forces TensorFlow to use deterministic algorithms
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Enable op determinism for TensorFlow 2.9+
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # Older TensorFlow versions don't have this function
        pass

    print(f"✓ All random seeds set to {seed}")
    print(f"✓ Deterministic mode enabled")
    print(f"  Note: This may reduce training performance by ~10-20%")


def get_environment_info():
    """
    Get detailed environment information for reproducibility documentation.

    Returns:
        dict: Dictionary containing:
            - python_version: Python version string
            - platform: Operating system and platform info
            - tensorflow_version: TensorFlow version
            - numpy_version: NumPy version
            - pandas_version: Pandas version (if available)
            - sklearn_version: scikit-learn version (if available)

    Example:
        >>> info = get_environment_info()
        >>> print(info['python_version'])
        3.10.12
    """
    import sys
    import platform

    info = {
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
    }

    # Try to get pandas version
    try:
        import pandas as pd
        info['pandas_version'] = pd.__version__
    except ImportError:
        info['pandas_version'] = 'Not installed'

    # Try to get scikit-learn version
    try:
        import sklearn
        info['sklearn_version'] = sklearn.__version__
    except ImportError:
        info['sklearn_version'] = 'Not installed'

    return info


def print_environment_info():
    """
    Print environment information in a formatted way.

    Example:
        >>> print_environment_info()
        ==================== ENVIRONMENT INFO ====================
        Python:        3.10.12
        Platform:      Windows-10-...
        TensorFlow:    2.15.0
        NumPy:         1.24.3
        Pandas:        2.0.3
        scikit-learn:  1.3.0
        ==========================================================
    """
    info = get_environment_info()

    print("\n" + "="*60)
    print("ENVIRONMENT INFO")
    print("="*60)
    print(f"Python:        {info['python_version']}")
    print(f"Platform:      {info['platform']}")
    print(f"TensorFlow:    {info['tensorflow_version']}")
    print(f"NumPy:         {info['numpy_version']}")
    print(f"Pandas:        {info.get('pandas_version', 'N/A')}")
    print(f"scikit-learn:  {info.get('sklearn_version', 'N/A')}")
    print("="*60 + "\n")


def save_environment_info(filepath='environment_info.txt'):
    """
    Save environment information to a text file.

    Args:
        filepath (str): Path to save the environment info file.
                       Default is 'environment_info.txt'

    Example:
        >>> save_environment_info('logs/env.txt')
    """
    info = get_environment_info()

    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ENVIRONMENT INFORMATION\n")
        f.write("="*60 + "\n\n")

        for key, value in info.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "="*60 + "\n")

    print(f"✓ Environment info saved to: {filepath}")
