
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameters: dict,
    random_state: int,
    n_iter: int = 50,
    cv: int = 5
) -> dict:
    """
    Train an XGBoost regressor with hyperparameter tuning using randomized grid search.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        X_test (pd.DataFrame): Testing feature data.
        y_test (pd.Series): Testing target data.
        hyperparameters (dict): Hyperparameter search space for XGBoost.
        random_state (int): Random state for reproducibility.
        n_iter (int): Number of parameter settings sampled. Defaults to 50.
        cv (int): Number of cross-validation folds. Defaults to 5.

    Returns:
        dict: A dictionary containing the best model, its parameters, and evaluation metrics.
    """
    # Define the base XGBoost model
    xgboost_model = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)

    # Perform randomized grid search
    random_search = RandomizedSearchCV(
        estimator=xgboost_model,
        param_distributions=hyperparameters,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=cv,
        random_state=random_state,
        verbose=1
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "root_mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred)),
        "best_params": random_search.best_params_
    }

    # Log results
    print("Best Parameters:", metrics["best_params"])
    print("Mean Squared Error:", metrics["mean_squared_error"])
    print("Root Mean Squared Error:", metrics["root_mean_squared_error"])

    # Return the trained model and metrics
    return {"model": best_model, "metrics": metrics}