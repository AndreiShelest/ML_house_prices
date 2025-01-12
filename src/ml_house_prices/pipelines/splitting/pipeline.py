from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_data

def create_splitting_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=split_data,
                inputs=[
                    'house_prices',
                    'params:random_state',
                    'params:splitting',
                ],
                outputs=[
                    'HP_train',
                    'HP_test',
                    'Y_train',
                    'Y_test',
                ],
                name='split_data_node',
                tags=["split_data"]
            )])