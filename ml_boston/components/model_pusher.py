import sys

from ml_boston.exception import ExceptionHandle
from ml_boston.logger import logging
from ml_boston.entity.artifact_entity import LocalModelPusherArtifact,LocalModelEvaluationArtifact
from ml_boston.entity.config_entity import LocalModelPusherConfig
from ml_boston.entity.s3_estimator import FraudDetectionLocalModelEstimator
from ml_boston.constants import *
import joblib


class LocalModelPusher:
    def __init__(self, model_evaluation_artifact: LocalModelEvaluationArtifact,
                 model_pusher_config: LocalModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.FraudDetection_estimator = FraudDetectionLocalModelEstimator(model_path=model_pusher_config.model_key_path)

    def initiate_model_pusher(self) -> LocalModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")
            model = joblib.load(self.model_evaluation_artifact.trained_model_path)
            self.FraudDetection_estimator.save_model(model)

            model_pusher_artifact = LocalModelPusherArtifact(model_path=self.model_pusher_config.model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise ExceptionHandle(e, sys) from e