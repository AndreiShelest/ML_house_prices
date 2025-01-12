from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  train_xgboost_regressor

def create_splitting_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=train_xgboost_regressor,
                inputs=[
                    'house_prices',
                    'params:random_state',
                    'params:splitting',
                ],
                outputs=[
                    'HP_train',
                    'HP_test',
                    'HP_y_train',
                    'HP_y_test',
                ],
                name='split_data_node',
                tags=["split_data"]
            )])