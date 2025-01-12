from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_date_column, extract_quarter, impute_categorical, continous_imputation, encode_one_hot


def create_data_processing_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
            func=extract_quarter,
            inputs=["HP_train", "HP_test", "params:quarter_column"],
            outputs=["x_train_quarter", "x_test_quarter"],
            name="extract_quarter_node"
        ),
        
        node(
            func=split_date_column,
            inputs=["x_train_quarter", "x_test_quarter", "params:date_column"],
            outputs=["x_train_date_split", "x_test_date_split"],
            name="split_date_column_node"
        ),
        
        node(
            func=impute_categorical,
            inputs=["x_train_date_split", "x_test_date_split", "params:categorical_columns", "params:exclude_columns", "params:random_state"],
            outputs=["x_train_categorical_imputed", "x_test_categorical_imputed"],
            name="impute_categorical_node"
        ),
        
        node(
            func=encode_one_hot,
            inputs=["x_train_categorical_imputed", "x_test_categorical_imputed", "params:encoding", "Y_train"],
            outputs=["x_train_encoded", "x_test_encoded"],
            name="encode_one_hot_node"
        ),
        
        node(
            func=continous_imputation,
            inputs=["x_train_encoded", "x_test_encoded", "params:continuous_imputation_params"],
            outputs=["X_train", "X_test"],
            name="continuous_imputation_node"
        ),
    ])