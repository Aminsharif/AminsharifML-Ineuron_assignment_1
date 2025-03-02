from ml_boston.constants import DATA_URL
from ml_boston.exception import ExceptionHandle
import pandas as pd
import sys
from typing import Optional
import numpy as np



class Data:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.data = pd.read_csv(DATA_URL) 
        except Exception as e:
            raise ExceptionHandle(e,sys)
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            
            """
            df = self.data
            return df
        except Exception as e:
            raise ExceptionHandle(e,sys)