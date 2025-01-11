from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def split_data(data: pd.DataFrame, model_options: dict, data_params: dict) -> tuple:
    target_col = data_params['target_col']
    data = data.drop(columns=data_params['column_to_drop'])

    X = data.drop(columns=target_col)
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_options['test_size'],
        random_state=model_options['random_state'],
        stratify=y,
    )

    return X_train, X_test, y_train, y_test