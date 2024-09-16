import pandas as pd
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)


class DataIngestion:
    """
    This class is fully responsible for data ingestion process from any source.
    """

    def __init__(self, data_path,output_path):
        self.data_path = data_path
        self.output_path=output_path

    def read_data(self):
        """
        This fun will read data from the given path and 
        """
        try:
            logging.info(f"Reading data from {self.data_path}")
            data=pd.read_csv(self.data_path)

            # save the data to the output path
            data.to_csv(self.output_path,index=False)
            logging.info(f"Data saved to {self.output_path}")

            return data
        
        except Exception as e:
            return logging.error(f"Error in reading the data: {e}")

if __name__=="__main__":
    data_path="Given_Data/Urban Air Quality and Health Impact Dataset.csv"
    output_path="data/raw/raw.csv"

    data_ingestion=DataIngestion(data_path,output_path) 
    data_ingestion.read_data()