"""
preprocessing.py

Contains reusable preprocessing utilities for the
AutoPrice AI Pro project.
"""

import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


def build_preprocessor(X):
    """
    Builds a preprocessing pipeline for numerical
    and categorical features.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    ColumnTransformer
    """

    numerical_columns = X.select_dtypes(
        include = np.number
    ).columns

    categorical_columns = X.select_dtypes(
        exclude = np.number
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num" , StandardScaler() , numerical_columns),
            ("cat" , OneHotEncoder(handle_unknown="ignore") , categorical_columns)
        ]
    )

    return preprocessor