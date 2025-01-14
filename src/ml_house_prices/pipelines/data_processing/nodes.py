import logging
import pandas as pd
import numpy as np
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from category_encoders.target_encoder import TargetEncoder

def remove_outliers(
    y_train: pd.Series, 
    x_train: pd.DataFrame, 
    y_test: pd.Series, 
    x_test: pd.DataFrame, 
    contamination: float = 0.025
) -> dict:

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(y_train.values.reshape(-1, 1))

    # Predict outliers for y_train and y_test
    train_outlier_labels = iso.predict(y_train.values.reshape(-1, 1))
    test_outlier_labels = iso.predict(y_test.values.reshape(-1, 1))

    # Filter out outliers
    x_train_clean = x_train[train_outlier_labels == 1]
    y_train_clean = y_train[train_outlier_labels == 1]
    x_test_clean = x_test[test_outlier_labels == 1]
    y_test_clean = y_test[test_outlier_labels == 1]

    return x_train_clean, y_train_clean, x_test_clean, y_test_clean
    
def log_transform(data, column_name):
    data[column_name] = np.log(data[column_name]) 
    return data
def log_transform_target(target):
    return np.log(target)

def extract_quarter(x_train, x_test, column):

    x_train = x_train.copy()
    x_test = x_test.copy()
    
    x_train[column] = x_train[column].str[-2:]
    x_test[column] = x_test[column].str[-2:]
    
    return x_train, x_test

def split_date_column(x_train, x_test, date_col):
    x_train = x_train.copy()
    x_test = x_test.copy()

    x_train[date_col] = pd.to_datetime(x_train[date_col], errors='coerce')
    x_test[date_col] = pd.to_datetime(x_test[date_col], errors='coerce')
    for df in [x_train, x_test]:
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
    x_train.drop(columns=[date_col], inplace=True)
    x_test.drop(columns=[date_col], inplace=True)

    return x_train, x_test

def dropping_columns(x_train, x_test, columns_to_drop):
    x_test = x_test.drop(columns=columns_to_drop)
    x_train = x_train.drop(columns=columns_to_drop)
    return x_train, x_test



# def encode_categorical(data: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
#     data = data.copy()
#     encoder = TargetEncoder(cols=[column])
#     data[column] = encoder.fit_transform(data[column], data[target])
    
#     return data

def impute_categorical(x_train: pd.DataFrame, x_test: pd.DataFrame, categorical_columns: list, exclude_cols: list, random_state: int):
    x_train = x_train.copy()
    x_test = x_test.copy()
    for column in categorical_columns:

        predictors = [col for col in x_train.columns if col not in exclude_cols + [column]]

        missing_mask_train = x_train[column].isna()
        if missing_mask_train.sum() > 0:

            train_data = x_train[~missing_mask_train]
            predict_data_train = x_train[missing_mask_train]
            clf = DecisionTreeClassifier(random_state=random_state)
            clf.fit(train_data[predictors], train_data[column])

            
            imputed_values_train = clf.predict(predict_data_train[predictors])
            x_train.loc[missing_mask_train, column] = imputed_values_train

        missing_mask_test = x_test[column].isna()
        if missing_mask_test.sum() > 0:
     
            imputed_values_test = clf.predict(x_test[missing_mask_test][predictors])
            x_test.loc[missing_mask_test, column] = imputed_values_test

    return x_train, x_test

def encode_one_hot(x_train: pd.DataFrame, x_test: pd.DataFrame, parameters: dict, target: pd.Series):
    catboost_columns = parameters.get("catboost", [])
    onehot_columns = parameters.get("one_hot", [])
    
    if not catboost_columns and not onehot_columns:
        raise ValueError("Both 'catboost_columns' and 'onehot_columns' cannot be empty.")

    x_train = x_train.copy()
    x_test = x_test.copy()

    # Perform CatBoost encoding
    if catboost_columns:
        catboost_encoder = CatBoostEncoder(cols=catboost_columns)
        catboost_encoder.fit(x_train[catboost_columns], target)
        x_train[catboost_columns] = catboost_encoder.transform(x_train[catboost_columns])
        x_test[catboost_columns] = catboost_encoder.transform(x_test[catboost_columns])

    # Perform One-Hot encoding
    if onehot_columns:
        combined = pd.concat([x_train[onehot_columns], x_test[onehot_columns]], axis=0, keys=["train", "test"])
        onehot_encoded = pd.get_dummies(combined, columns=onehot_columns, drop_first=True)

        x_train_encoded = onehot_encoded.xs("train")
        x_test_encoded = onehot_encoded.xs("test")
        
        x_train = x_train.drop(columns=onehot_columns).join(x_train_encoded)
        x_test = x_test.drop(columns=onehot_columns).join(x_test_encoded)

    return x_train, x_test

def continous_imputation(x_train, x_test, params, random_state):
    exclude_cols = params["exclude_cols"]
    target_cols = params["target_col"]
    
 
    for column in target_cols:

        predictors = [
            col for col in x_train.columns if col not in exclude_cols + [column]
        ]

        train_non_missing = x_train[~x_train[column].isnull()]
        train_missing = x_train[x_train[column].isnull()]
        test_missing = x_test[x_test[column].isnull()]
        

        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=random_state)
        model.fit(train_non_missing[predictors], train_non_missing[column])
        
 
        if not train_missing.empty:
            x_train.loc[train_missing.index, column] = model.predict(train_missing[predictors])
        if not test_missing.empty:
            x_test.loc[test_missing.index, column] = model.predict(test_missing[predictors])
    
    return x_train, x_test

def sanitize_feature_names(x_train, x_test):

    def clean_columns(df):
        df.columns = df.columns.str.replace(r"[^\w\s]", "", regex=True)
        return df
    
    x_train = clean_columns(x_train)
    x_test = clean_columns(x_test)
    return x_train, x_test