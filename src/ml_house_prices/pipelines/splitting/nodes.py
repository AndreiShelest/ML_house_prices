from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
def split_data(data: pd.DataFrame, random_state: int, data_params: dict) -> tuple:
    target_col = data_params['target_column']
    data = data.drop(columns=data_params['column_to_drop'])
    X = data.drop(columns=target_col)
    y = data[target_col]

    # Perform train-test split without stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_params['test_size'],
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test