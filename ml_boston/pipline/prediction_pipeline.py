import sys
from ml_boston.entity.config_entity import BostonPredictorConfig
from ml_boston.entity.s3_estimator import BostonLocalModelEstimator
from ml_boston.exception import ExceptionHandle
from ml_boston.logger import logging
from ml_boston.entity.config_entity import LocalModelPusherConfig
import joblib
from pandas import DataFrame

class BostonData:
    def __init__(self,
                crim,
                zn,
                indus,
                chas,
                nox,
                rm,
                age,
                dis,
                rad,
                tax,
                ptratio,
                b,
                lstat               
            ):
        """
        boston Data constructor
        Input: all features of the trained model for prediction
        """
        
        try:
            self.crim = crim
            self.zn= zn
            self.indus= indus
            self.chas= chas
            self.nox= nox
            self.rm= rm
            self.age= age
            self.dis= dis
            self.rad= rad
            self.tax= tax
            self.ptratio= ptratio
            self.b= b
            self.lstat= lstat
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def get_boston_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from bostonData class input
        """
        try:
            
            boston_input_dict = self.get_boston_data_as_dict()
            return DataFrame(boston_input_dict)
        
        except Exception as e:
            raise ExceptionHandle(e, sys) from e


    def get_boston_data_as_dict(self):
        """
        This function returns a dictionary from bostonData class input 
        """
        logging.info("Entered get_boston_data_as_dict method as bostonData class")

        try:
            input_data = {
                "crim": [self.crim],
                "zn": [self.zn],
                "indus": [self.indus],
                "nox": [self.nox],
                "rm": [self.rm],
                "age": [self.age],
                "dis": [self.dis],
                "rad": [self.rad],
                "tax": [self.tax],
                "ptratio": [self.ptratio],
                "b": [self.b],
                "lstat": [self.lstat],
                "chas": [self.chas],
            }

            logging.info("Created boston data dict")

            logging.info("Exited get_boston_data_as_dict method as bostonData class")

            return input_data

        except Exception as e:
            raise ExceptionHandle(e, sys) from e
        
class BostonClassifier:
    def __init__(self,prediction_pipeline_config: BostonPredictorConfig = BostonPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise ExceptionHandle(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of BostonClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of BostonClassifier class")
            model = BostonLocalModelEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise ExceptionHandle(e, sys)
        
class BostonClassifierWithLocalModel:
    def __init__(self,local_model_pusher: LocalModelPusherConfig = LocalModelPusherConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.local_model_pusher_artifact = local_model_pusher
        except Exception as e:
            raise ExceptionHandle(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of BostonClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of BostonClassifier class")
            model = joblib.load(self.local_model_pusher_artifact.model_key_path)
            result =  model.predict(dataframe)
            print('**********************************', result)
            return result
        
        except Exception as e:
            logging.info(e)
            raise ExceptionHandle(e, sys)