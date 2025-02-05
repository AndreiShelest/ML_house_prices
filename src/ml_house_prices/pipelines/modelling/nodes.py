from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from catboost import CatBoostRegressor
import yaml
from sklearn.preprocessing import StandardScaler

def parse_distribution(dist_string):
    if dist_string.startswith("randint"):
        args = eval(dist_string[len("randint("):-1])
        return randint(*args)
    elif dist_string.startswith("uniform"):
        args = eval(dist_string[len("uniform("):-1])
        return uniform(*args)
    else:
        raise ValueError(f"Unsupported distribution: {dist_string}")


def train_benchmark(x_train, y_train, param_grid, cv_folds, random_state):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    updated_param_grid = {
        "model__fit_intercept": param_grid.get("fit_intercept", [True]),
        "model__positive": param_grid.get("positive", [False]),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=updated_param_grid,
        cv=cv_folds,
        n_iter=1,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"].tolist()

    return best_model, best_params, cv_scores


def evaluate_model(X_test, y_test, model, model_name, inverse_transform=np.exp):
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()

    predictions = model.predict(X_test)

    if predictions.ndim > 1:
        predictions = predictions.squeeze()

    predictions = pd.Series(predictions, index=y_test.index)

    if inverse_transform:
        max_log_value = 700  
        predictions = np.clip(predictions, a_min=None, a_max=max_log_value)
        predictions = inverse_transform(predictions)

        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            raise ValueError("Predictions contain NaN or infinite values after inverse transformation.")

    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
        raise ValueError("y_test contains NaN or infinite values.")

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - predictions)) 
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)

   
    n = len(y_test)  
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"Model: {model_name}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"R2: {r2}")
    print(f"Adjusted R2: {adjusted_r2}")

    return {
        "model": model_name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "adjusted_r2": adjusted_r2
    }


def train_xgboost(x_train, y_train, param_grid, cv_folds, random_state):
    parsed_param_grid = {key: parse_distribution(value) for key, value in param_grid.items()}

    model = XGBRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=parsed_param_grid,
        cv=cv_folds,
        n_iter=20,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"].tolist()

    return best_model, best_params, cv_scores


def train_lightgbm(x_train, y_train, param_grid, cv_folds, random_state):
    parsed_param_grid = {key: parse_distribution(value) for key, value in param_grid.items()}

    model = LGBMRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=parsed_param_grid,
        cv=cv_folds,
        n_iter=20,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"].tolist()

    return best_model, best_params, cv_scores


def train_catboost(x_train, y_train, param_grid, cv_folds, random_state):
    parsed_param_grid = {key: parse_distribution(value) for key, value in param_grid.items()}

    model = CatBoostRegressor(
        random_seed=random_state,
        silent=True
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parsed_param_grid,
        cv=cv_folds,
        n_iter=20,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"].tolist()

    return best_model, best_params, cv_scores