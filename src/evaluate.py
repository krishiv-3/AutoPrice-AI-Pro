"""
evaluate.py

Contains reusable evaluation functions
for regression models.
"""


import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model , X_test , y_test):
    """
    Evaluate a trained regression model.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained machine learning pipeline.

    X_test : pandas.DataFrame
        Test features.

    y_test : pandas.Series
        Actual target values.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    y_pred = model.predict(X_test)

    metrics  = {
        "MAE" : mean_absolute_error(y_test , y_pred),
        "MSE" : mean_squared_error(y_test,y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test,y_pred)),
        "R2" : r2_score(y_test,y_pred)
    }

    return metrics

def print_metrics(metrics):
    """
    Display evaluation metrics in a readable format.
    """

    print("=" * 40)
    print("Regression Model Performance")
    print("=" * 40 )

    print(f"MAE : {metrics["MAE"]:.2f}")
    print(f"MSE : {metrics["MSE"]:.2f}")
    print(f"RMSE : {metrics["RMSE"]:.2f}")
    print(f"R2 : {metrics["R2"]:.2f}")

    print("=" * 40 )