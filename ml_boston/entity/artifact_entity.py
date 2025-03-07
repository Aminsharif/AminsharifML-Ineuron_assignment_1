from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str 

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str


@dataclass
class ClassificationMetricArtifact:
    mse:float
    rmse:float
    r2:float



@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact




@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str

@dataclass
class LocalModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    trained_model_path:str



@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str

@dataclass
class LocalModelPusherArtifact:
    model_path:str
