from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_date_column, extract_quarter, impute_categorical, continous_imputation, encode_one_hot, sanitize_feature_names, dropping_columns, remove_outliers

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=remove_outliers,
            inputs=["HP_y_train", "HP_train","HP_y_test", "HP_test"],
            outputs=["x_train_clean", "Y_train", "x_test_clean", "Y_test"],
            name="remove_outliers_node"
        ),
        node(
            func=extract_quarter,
            inputs=["x_train_clean", "x_test_clean", "params:quarter_column"],
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
            func=dropping_columns,
            inputs=["x_train_date_split", "x_test_date_split", "params:columns_to_drop"],
            outputs=["x_train_dropped", "x_test_dropped"],
            name="dropping_columns_node"
        ),
        
        
        # Perform encoding earlier in the pipeline
        node(
            func=encode_one_hot,
            inputs=["x_train_dropped", "x_test_dropped", "params:encoding", "Y_train"],
            outputs=["x_train_encoded", "x_test_encoded"],
            name="encode_one_hot_node"
        ),
        

        node(
            func=impute_categorical,
            inputs=["x_train_encoded", "x_test_encoded", "params:categorical_columns", "params:exclude_columns", "params:random_state"],
            outputs=["x_train_categorical_imputed", "x_test_categorical_imputed"],
            name="impute_categorical_node"
        ),
        
        # Perform continuous imputation as the final step
        node(
            func=continous_imputation,
            inputs=["x_train_categorical_imputed", "x_test_categorical_imputed", "params:continuous_imputation_params", "params:random_state"],
            outputs=["X_train_almost_ready", "X_test_almost_ready"],
            name="continuous_imputation_node"
        ),
        node(
            func=sanitize_feature_names,
            inputs=["X_train_almost_ready", "X_test_almost_ready"],
            outputs=["X_train", "X_test"],
            name="sanitize_feature_names_node"
        )
    ])

