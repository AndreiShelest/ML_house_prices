from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# def random_grid_search(x_train, y_train, model_type, param_grid, cv_folds, random_state):

#     model = model_type()
#     search = RandomizedSearchCV(
#         model, param_grid, cv=cv_folds, n_iter=20, random_state=random_state, scoring="neg_mean_squared_error"
#     )
#     search.fit(x_train, y_train)
    
#     best_model = search.best_estimator_
#     best_params = search.best_params_
#     cv_scores = search.cv_results_['mean_test_score']

#     return best_model, best_params, cv_scores


def train_benchmark(x_train, y_train, param_grid, cv_folds, random_state):
    model = LinearRegression()
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        cv=cv_folds,
        n_iter=1,  
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)
    

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"]


    return best_model, best_params, cv_scores

def evaluate_model(x_test, y_test, model, model_name):
    predictions = model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)


    return {
        "model": model_name,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def train_xgboost(x_train, y_train, param_grid, cv_folds, random_state):
    model = XGBRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        cv=cv_folds,
        n_iter=20,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"]

    return best_model, best_params, cv_scores

def train_lightgbm(x_train, y_train, param_grid, cv_folds, random_state):
    model = LGBMRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        cv=cv_folds,
        n_iter=20,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_scores = search.cv_results_["mean_test_score"]

    return best_model, best_params, cv_scores