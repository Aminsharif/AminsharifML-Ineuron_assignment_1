import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from ml_boston.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from ml_boston.entity.config_entity import DataTransformationConfig
from ml_boston.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from ml_boston.exception import ExceptionHandle
from ml_boston.logger import logging
from ml_boston.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from sklearn.impute import SimpleImputer


class DataTransformation():
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact=None,
                 data_transformation_config: DataTransformationConfig=None,
                 data_validation_artifact: DataValidationArtifact=None):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise ExceptionHandle(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ExceptionHandle(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            preprocessor = ColumnTransformer(
                [
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise ExceptionHandle(e, sys) from e
        

    def is_null_present(self,data):
        """
            Method Name: is_null_present
            Description: This method checks whether there are null values present in the pandas Dataframe or not.
            Output: Returns True if null values are present in the DataFrame, False if they are not present and
                returns the list of columns for which null values are present.
             On Failure: Raise Exception
        """
        logging.info('Entered the is_null_present method of the DataTransformation class')
        null_present = False
        cols_with_missing_values=[]
        cols = data.columns
        try:
            null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(null_counts)):
                if null_counts[i]>0:
                    null_present=True
                    cols_with_missing_values.append(cols[i])
            if(null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
            logging.info('Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the DataTransformation class')
            return null_present, cols_with_missing_values
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def impute_missing_values(self, data, cols_with_missing_values):
        """
             Method Name: impute_missing_values
             Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
             Output: A Dataframe which has all the missing values imputed.
             On Failure: Raise Exception
        """

        logging.info(
            "Entered the impute_missing_values method of DataTransformation class"
        )

        try:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            for col in cols_with_missing_values:
                data[col] = imputer.fit_transform (data[[col]]).reshape(-1)
            logging.info('Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return data
        except Exception as e:
            raise ExceptionHandle(e, sys) from e
    
    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                drop_cols = self._schema_config['drop_columns']

                logging.info("drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)
                
                is_null_present_train_df, cols_with_missing_values_train_df = self.is_null_present(input_feature_train_df)
                if is_null_present_train_df:
                    input_feature_train_df = self.impute_missing_values(input_feature_train_df, cols_with_missing_values_train_df)

                logging.info("Got train features and test features of Training dataset")
                

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)


                is_null_present_test_df, cols_with_missing_values_test_df = self.is_null_present(input_feature_test_df)
                if is_null_present_test_df:
                    input_feature_test_df = self.impute_missing_values(input_feature_test_df, cols_with_missing_values_test_df)

                logging.info("drop the columns in drop_cols of Test dataset")

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                train_arr = np.c_[
                    np.array(input_feature_train_df), np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    np.array(input_feature_test_df), np.array(target_feature_test_df)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise ExceptionHandle(e, sys) from e