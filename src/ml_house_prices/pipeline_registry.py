"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from ml_house_prices.pipelines.splitting import pipeline as splitting_pipeline
from ml_house_prices.pipelines.data_processing import pipeline as data_processing_pipeline
from ml_house_prices.pipelines.modelling import pipeline as modelling_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    splitting = splitting_pipeline.create_splitting_pipeline()
    data_processing = data_processing_pipeline.create_data_processing_pipeline()
    modelling = modelling_pipeline.create_pipeline_model

    # Register pipelines
    return {
        "__default__": splitting + data_processing + modelling,
        "data_splitting_pipeline": splitting,       
        "data_processing_pipeline": data_processing,
        "modelling_pipeline": modelling,
    }
