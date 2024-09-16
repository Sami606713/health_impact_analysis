from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)

class FeatureEngineering:
    def __init__(self,data_path,output_path):
        self.data_path=data_path
        self.output_path=output_path
        self.data=pd.read_csv(data_path)
    
    
    def extract_feature(self,df):
        """
        This function is responsible for extracting the feature from the dataset.
        """
        try:
            logging.info("Extracting Year month and day from datatime col")
            if "datetime" in df.columns:
                df['year'] = pd.to_datetime(df['datetime']).dt.year
                df['Month']= pd.to_datetime(df['datetime']).dt.month_name()
                df['Day_of_Week']= pd.to_datetime(df['datetime']).dt.day_name()
            
            logging.info("Extracting hour and minute from sunset col")
            if "sunset" in df.columns:
                df['sunset'] = pd.to_datetime(df['sunset'])
                df['sunset_hour'] = df['sunset'].dt.hour
                df['sunset_minute'] = df['sunset'].dt.minute
            
            logging.info("Extracting hour and minute from sunrise col")
            if "sunrise" in df.columns:
                df['sunrise'] = pd.to_datetime(df['sunrise'])
                df['sunrise_hour'] = df['sunrise'].dt.hour
                df['sunrise_minute'] = df['sunrise'].dt.minute
            
            
        except Exception as e:
            logging.error(f"Error in reading the data: {e}")
            raise


    def drop_unwanted_columns(self,df):
        """
        This function will drop the unwanted columns based on our knowledge from the dataset.
        we can also drop those columns whose value shold not be change.
        """
        try:
            drop_col_list=['datetime','sunset','sunrise']
            # get more unwanted columns make a list of categorical columns
            cat_col_list=["conditions","description","icon","source","City","Month","Season","Day_of_Week",'year']
            for col in cat_col_list:
                curr_val=df[col].value_counts().index
                if len(curr_val)==1:
                    logging.info(f"Column Name: {col} unique values: {curr_val}")
                    drop_col_list.append(col)
            
            logging.info(f"Columns to drop: {drop_col_list}")
            df=df.drop(columns=drop_col_list)
            return df

        except Exception as e:
            logging.error(f"Error in reading the data: {e}")
            raise

    def apply_vif(self,df):
        """
        1-This function will calculate the variance inflence factor for the dataset.
        2- Basically it will check the multicollinearity between the features.
        3- Generally value of VIF should be less then 5.
        4- If VIF value is greater then 5 then we can drop that column.
        """
        try:
            num_col=df.select_dtypes(include=np.number).columns

            col_to_drop=[]
            vif_data=df[num_col]
            column_index=0
            for i in range(len(num_col)):
                vif_value=variance_inflation_factor(vif_data.values,column_index)
                logging.info(f"{num_col[i]}=====>{vif_value}")
                if vif_value<5:
                    col_to_drop.append(num_col[i])
                    column_index+=1
                else:
                    vif_data=vif_data.drop(columns=[num_col[i]])
            
            logging.info(f"Columns to drop using Variance Inflence Factor: {col_to_drop}")
            df=df.drop(columns=col_to_drop)
            return df
        except Exception as e:
            logging.error(f"Error in reading the data: {e}")
            raise

    def process(self):
        """
        This fun will process the data and save the data to the output path.
        """
        try:
            df=self.data
            # extrcat the relavant feature
            self.extract_feature(df)

            # drop unwanted columns
            df=self.drop_unwanted_columns(df)

            # apply VIF
            df=self.apply_vif(df)

            # save the data
            df.to_csv(self.output_path,index=False)
        except Exception as e:
            logging.error(f"Error in reading the data: {e}")
            raise

if __name__=="__main__":
    data_path='data/raw/clean.csv'
    output_path='data/processed/process.csv'

    logging.info("Feature Engineering Started......")
    fe=FeatureEngineering(data_path=data_path,output_path=output_path)
    fe.process()
    logging.info("Feature Engineering Completed......")