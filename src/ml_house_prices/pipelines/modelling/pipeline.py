from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model, train_benchmark, train_lightgbm, train_xgboost


def create_pipeline_model(**kwargs):
    return Pipeline(
        [
            node(
                func=train_lightgbm,
                inputs=["X_train", "Y_train", "params:lightgbm_model.param_grid", "params:cv_folds", "params:random_state"],
                outputs=["lightgbm_model", "lightgbm_hyperparameters", "lightgbm_cv_scores"],
                name="train_lightgbm_node",
            ),
            node(
                func=train_xgboost,
                inputs=["X_train", "Y_train", "params:xgboost_model.param_grid", "params:cv_folds", "params:random_state"],
                outputs=["xgboost_model", "xgboost_hyperparameters", "xgboost_cv_scores"],
                name="train_xgboost_node",
            ),
            node(
                func=train_benchmark,
                inputs=["X_train", "Y_train", "params:linear_regression_model.param_grid", "params:cv_folds", "params:random_state"],
                outputs=["linear_regression_model", "linear_regression_hyperparameters", "linear_regression_cv_scores"],
                name="train_linear_regression_node",
            ),

            node(
                func=evaluate_model,
                inputs=["X_test", "Y_test", "lightgbm_model", "params:model_names.lightgbm"],
                outputs="lightgbm_test_scores",
                name="evaluate_lightgbm_node",
            ),
            node(
                func=evaluate_model,
                inputs=["X_test", "Y_test", "xgboost_model", "params:model_names.xgboost"],
                outputs="xgboost_test_scores",
                name="evaluate_xgboost_node",
            ),

            node(
                func=evaluate_model,
                inputs=["X_test", "Y_test", "linear_regression_model", "params:model_names.linear_regression"],
                outputs="linear_regression_test_scores",
                name="evaluate_linear_regression_node",
            ),
        ]
    )