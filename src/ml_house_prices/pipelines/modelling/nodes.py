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
        args = eval(dist_string[len("randint("):-1])  # Extract arguments
        return randint(*args)
    elif dist_string.startswith("uniform"):
        args = eval(dist_string[len("uniform("):-1])  # Extract arguments
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
    cv_scores = search.cv_results_["mean_test_score"].tolist()  # Convert to list for JSON serialization

    return best_model, best_params, cv_scores

def evaluate_model(X_test, y_test, model, model_name):

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()  # Convert DataFrame with one column to Series

    predictions = model.predict(X_test)
    
    if predictions.ndim > 1:  
        predictions = predictions.squeeze() 
    
    predictions = pd.Series(predictions, index=y_test.index)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)


    print(f"Model: {model_name}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}")
    print(f"R2: {r2}")

    return {
        "model": model_name,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }


def train_xgboost(x_train, y_train, param_grid, cv_folds, random_state):

    parsed_param_grid = {key: parse_distribution(value) for key, value in param_grid.items()}

    model = XGBRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=parsed_param_grid,
        cv=cv_folds,
        n_iter=10,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"].tolist()  # Convert to list for JSON serialization

    return best_model, best_params, cv_scores

def train_lightgbm(x_train, y_train, param_grid, cv_folds, random_state):
    parsed_param_grid = {key: parse_distribution(value) for key, value in param_grid.items()}

    model = LGBMRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=parsed_param_grid,
        cv=cv_folds,
        n_iter=10,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"].tolist()  # Convert to list for JSON serialization

    return best_model, best_params, cv_scores


def train_catboost(x_train, y_train, param_grid, cv_folds, random_state):
    parsed_param_grid = {key: parse_distribution(value) for key, value in param_grid.items()}

    model = CatBoostRegressor(
        random_seed=random_state,
        silent=True  # Silence training logs for RandomizedSearchCV
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parsed_param_grid,
        cv=cv_folds,
        n_iter=10,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    # Extract results
    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"].tolist()  # Convert to list for JSON serialization

    return best_model, best_params, cv_scores