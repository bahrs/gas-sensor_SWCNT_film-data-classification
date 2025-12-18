"""
seed.py

Global random seed control for reproducibility.
"""

import os
import random
import numpy as np


def set_global_seed(seed: int, deterministic_tf: bool = False) -> None:
    """
    Set global random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic_tf: Enable TensorFlow deterministic ops (may reduce performance)
    
    Note:
        Optuna uses its own internal RNG, so trials won't be 100% deterministic
        across runs, but model training will be reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        if deterministic_tf:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                # Not all TF versions support this
                pass
    except ImportError:
        pass  # TensorFlow not installed