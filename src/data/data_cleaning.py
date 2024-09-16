# Data Cleaning Script
import pandas as pd
import numpy as np
import logging
import os
logging.basicConfig(level=logging.INFO)

class DataCleaning:
    """
    This class is fully responsible for data cleaning process.
    0- Handling DataTypes
    1- Handling Null Values
    2- Removing Duplicates
    3- Handling Outliers
    4- Remove Unwanted Columns
    """
    def __init__(self,data_path,output_path):
        self.data_path=data_path
        self.output_path=output_path
        self.data=pd.read_csv(data_path)

    def handle_null_values(self,df):
        """
        This function will handle the null values in the dataset
        1- First we will itrate over the columns and check the missing values on each col.
        2- If any column contain more missing values we can drop those columns
        3- We can also check if missing value ratio is less then 5% we can drop the missing values oherwise we can fill the values according to distrubution.
        """
        try:
            logging.info("Handling the missing values")
            col_to_drop=[]
            for col in df.columns:
                null_ratio=df[col].isnull().mean()*100
                if null_ratio>0 and null_ratio<=5:
                    df.dropna(subset=[col], inplace=True)
                elif null_ratio>5 and null_ratio<=10:
                    if df[col].dtype==object:
                        df[col]=df[col].fillna("missing")
                    else:
                        col_mean=df[col].mean()
                        df[col].fillna(col_mean,inplace=True)
                elif null_ratio>10:
                    col_to_drop.append(col)

            if col_to_drop:
                # drop the col
                logging.info(f"Dropped columns with too many missing values: {col_to_drop}")
                df=df.drop(columns=col_to_drop)
                return df
            else:
                logging.info("No columns to drop")
        except Exception as e:
            logging.error(f"Error in handling missing values: {e}")
            raise
        

    def handle_data_types(self,df):
        """
        This function will handle the data types of the columns
        1- Change datetime col to datatime format
        """
        try:
            logging.info("Handling the data types")
            if "datetime" in df.columns:
                df['datetime']=pd.to_datetime(df['datetime'])
                return df
            else:
                logging.info("No datetime column found")
        except Exception as e:
            logging.error(f"Error in reading the data: {e}")
            raise

    def handle_duplicates(self,df):
        """
        This function will handle the duplicates in the dataset
        
        pandas.DataFrame.drop_duplicates
        """
        try:
            logging.info("Handling the duplicates")
            df=df.drop_duplicates()
            return df
        except Exception as e:
            logging.error(f"Error in reading the data: {e}")
            raise
    
    def process(self):
        """
        This function will process all the data cleaning steps that we will define above
        """
        try:
            logging.info("Data Cleaning Started......")
            df=self.data
            # handle datatypes
            df=self.handle_data_types(df)

            # handle missing values
            df=self.handle_null_values(df)

            # handle duplicates
            df=self.handle_duplicates(df)

            logging.info("Data Cleaning Completed......")
            # save the data to the output path
            df.to_csv(self.output_path,index=False)
        except Exception as e:
            logging.error(f"Error in processing the data: {e}")
            raise


if __name__=="__main__":
    # set the path
    data_path="data/raw/raw.csv"
    output_path="data/raw/clean.csv"

    df_clean=DataCleaning(data_path=data_path,output_path=output_path)
    
    df_clean.process()
