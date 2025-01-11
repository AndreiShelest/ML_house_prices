from kedro.pipeline import Pipeline, node, pipeline

from .nodes import encode


def create_data_processing_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=encode,
                inputs=["HP_train", "params:processing: encoding", "HP_y_train"],
                outputs=["data"],
                name="encode",
            )]
    )