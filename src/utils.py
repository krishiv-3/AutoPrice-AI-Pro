"""
utils.py

Utility functions for AutoPrice AI Pro.
"""

import joblib 
from pathlib import Path

def save_model(model, file_path):
    """
    Save a trained model or pipeline.

    Parameters
    ----------
    model : object
        Trained model or pipeline.

    file_path : str
        Destination path.
    """

    Path(file_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model,file_path)

def load_model(file_path):
    """
    Load a saved model or pipeline.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    object
    """
    return joblib.load(file_path)