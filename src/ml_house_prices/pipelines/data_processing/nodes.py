import logging
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def encode(data: pd.DataFrame, parameters: dict, target: pd.Series):
    catboost_columns= parameters["catboost"]
    onehot_columns =parameters["one_hot"]
    if not catboost_columns and not onehot_columns:
        raise ValueError("Both catboost_columns and onehot_columns cannot be empty.")

    encoded_data = data.copy()

    # Perform CatBoost encoding
    if catboost_columns:
        catboost_encoder = CatBoostEncoder(cols=catboost_columns)
        encoded_catboost = catboost_encoder.fit_transform(data[catboost_columns], target)
        # Replace the original columns with the encoded ones
        encoded_data[catboost_columns] = encoded_catboost

    # Perform One-Hot encoding
    if onehot_columns:
        onehot_encoded = pd.get_dummies(data[onehot_columns], columns=onehot_columns, drop_first=True)
        # Drop the original onehot columns
        encoded_data = encoded_data.drop(columns=onehot_columns)
        # Concatenate the one-hot encoded columns
        encoded_data = pd.concat([encoded_data, onehot_encoded], axis=1)

    return encoded_data



def train_benchmark(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
 
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
    }

    # Log the evaluation metrics
    print(f"Linear Regression Evaluation Metrics:\n{metrics}")

    # Return the trained model and metrics
    return {"model": model, "metrics": metrics}

def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)