
from ml_boston.exception import ExceptionHandle
from ml_boston.entity.estimator import FraudDetectionModel
import sys
from pandas import DataFrame
import joblib
from ml_boston.utils.main_utils import save_object
import os

class BostonLocalModelEstimator:
    """
    This class is used to save and retrieve ml_bostons model in s3 bucket and to do prediction
    """

    def __init__(self,model_path):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """

        self.model_path = model_path
        self.loaded_model:FraudDetectionModel=None


    def is_model_present(self,model_path):
        try:
            if os.path.exists(model_path):
                return True
            else:
                return False
        except ExceptionHandle as e:
            print(e)
            return False

    def load_model(self,):
        """
        Load the model from the model_path
        :return:
        """

        return joblib.load(self.model_path)

    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            save_object(self.model_path , from_file)

        except Exception as e:
            raise ExceptionHandle(e, sys)


    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise ExceptionHandle(e, sys)