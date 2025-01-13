from kedro.pipeline import Pipeline
from ml_house_prices.pipelines.splitting.pipeline import create_pipeline as splitting_pipeline
from ml_house_prices.pipelines.data_processing.pipeline import create_pipeline as data_processing_pipeline
from ml_house_prices.pipelines.modelling.pipeline import create_pipeline as modelling_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    splitting = splitting_pipeline() 
    data_processing = data_processing_pipeline()
    modelling = modelling_pipeline()  

    return {
        "__default__": splitting + data_processing + modelling,
        "data_splitting_pipeline": splitting,
        "data_processing_pipeline": data_processing,
        "modelling_pipeline": modelling,
    }
