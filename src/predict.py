"""
predict.py

Contains reusable functions for loading a trained
pipeline and making predictions.
"""

import joblib 
import pandas as pd 

def load_pipeline(model_path):
    """
    Load a trained machine learning pipeline.

    Parameters
    ----------
    model_path : str
        Path to the saved pipeline.

    Returns
    -------
    Pipeline
    """

    pipeline = joblib.load(model_path)

    return pipeline

def predict_price(pipeline , new_data):
    """
    Predict car price.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Trained pipeline.

    new_data : pandas.DataFrame
        New data for prediction.

    Returns
    -------
    numpy.ndarray
    """

    prediction = pipeline.predict(new_data)

    return round(float(prediction[0]), 2)     