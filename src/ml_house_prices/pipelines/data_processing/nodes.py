import logging
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder

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

def continous_imputation(x_train, x_test, params):
    exclude_cols = params["exclude_cols"]
    target_col = params["target_col"]
    predictors = [col for col in x_train.columns if col not in exclude_cols + [target_col]]
    train_non_missing = x_train[~x_train[target_col].isnull()]
    train_missing = x_train[x_train[target_col].isnull()]

    model = LinearRegression()
    model.fit(train_non_missing[predictors], train_non_missing[target_col])
    if not train_missing.empty:
        x_train.loc[x_train[target_col].isnull(), target_col] = model.predict(train_missing[predictors])
    if x_test[target_col].isnull().any():
        x_test.loc[x_test[target_col].isnull(), target_col] = model.predict(x_test[x_test[target_col].isnull()][predictors])

    return x_train, x_test