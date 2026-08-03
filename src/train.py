"""
train.py

Contains reusable functions for training
machine learning models.
"""

from sklearn.pipeline import Pipeline

def train_model(preprocessor , model , X_train , y_train):
    """
    Train a machine learning pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Data preprocessing pipeline.

    model : sklearn estimator
        Any Scikit-learn regression model.

    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training target.

    Returns
    -------
    Pipeline
        Trained Scikit-learn pipeline.
    """

    pipeline = Pipeline([
        ("preprocessor" , preprocessor),
        ("model" , model)
    ])

    pipeline.fit(X_train,y_train)

    return pipeline