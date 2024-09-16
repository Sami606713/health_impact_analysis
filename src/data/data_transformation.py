from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from src.utils import save_model    
import pickle as pkl
import pandas as pd
import numpy as np
import logging
import os
logging.basicConfig(level=logging.INFO)

class DataTransformation:
    def __init__(self,data_path,train_output_path,test_output_path,transformer_path):
        self.train_output_path=train_output_path
        self.test_output_path=test_output_path
        self.transformer_path=transformer_path
        self.data=pd.read_csv(data_path)
    
    def split_data(self):
        """
        This function will split the data into train and test
        """
        try:
            logging.info("Saperate Feature and labels")
            feature=self.data.drop(columns='Health_Risk_Score')
            label=self.data['Health_Risk_Score']

            logging.info("Splitting the data into train and test")
            x_train,x_test,y_train,y_test = train_test_split(feature ,label ,test_size=0.2, random_state=42)
            return x_train,x_test,y_train,y_test
        except Exception as e:
            logging.error(f"Error in splitting the data: {e}")
            raise
    
    def build_pipelines(self,features):
        """
        This function will build the pipeline for the data transformation
        """
        try:
            logging.info("Saperate numerical categorical and text columns")
            num_col=features.select_dtypes(include='number').columns
            cat_col=features.drop(columns=['description']).select_dtypes(include=object).columns
            text_col=['description']

            logging.info("Building the num_pipeline")
            num_pipe=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scale",StandardScaler())
            ])

            logging.info("Building the cat_pipeline")
            cat_pipe=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder(sparse_output=False,drop="first",handle_unknown="ignore"))
            ])

            logging.info("Building the text_pipeline")
            text_transformers = [(col, TfidfVectorizer(max_features=500), col) for col in text_col]

            logging.info("Building the transformer")
            transformer=self.build_transformer(
                num_col=num_col,cat_col=cat_col,
                num_pipe=num_pipe,cat_pipe=cat_pipe,
                text_transformers=text_transformers
            )

            logging.info(f"saving the transformer at {self.transformer_path}")
            # self.save_model(transformer,self.transformer_path)
            return transformer

        except Exception as e:
            logging.error(f"Error in building the pipeline: {e}")
            raise


    def build_transformer(self,
                          num_pipe,cat_pipe,
                          num_col,cat_col,text_transformers):
        """
        This function will build the column transformer
        """
        try:
            processor=ColumnTransformer(transformers=[
                ("num_transform",num_pipe,num_col),
                ("cat_transform",cat_pipe,cat_col),
                *text_transformers
            ])

            return processor
        except Exception as e:
            logging.error(f"Error in building the transformer: {e}")
            raise
    
    def save_model(self,model,output_path):
        """
        This function will save the model to the disk
        """
        try:
            logging.info("Saving the model")
            with open(output_path, 'wb') as file:
                pkl.dump(model, file)
        except Exception as e:
            logging.error(f"Error in saving the model: {e}")
            raise
        
    def process(self):
        """
        This function will process the data and save the data to the output path.
        """
        try:
            logging.info("Data Transformation Started......")
            df=self.data
            # split the data
            x_train,x_test,y_train,y_test=self.split_data()
            # build the pipeline
            transformer=self.build_pipelines(features=x_train)

            # transform the data
            x_train_transform=transformer.fit_transform(x_train)
            x_test_tranform=transformer.transform(x_test)

            # concat the data
            logging.info("Concate the train and test array")
            train_array=np.c_[x_train_transform,y_train]
            test_array=np.c_[x_test_tranform,y_test]

            # logging.info("Save the train and test array")
            np.save(file=self.train_output_path,arr=train_array)
            np.save(file=self.test_output_path,arr=test_array)

            logging.info("Data Transformation Completed......")
        except Exception as e:
            logging.info(f"Error in processing the data: {e}")
            raise


if __name__=="__main__":
    # set the paths data_path,train_output_path,test_output_path,transformer_path
    data_path='data/processed/process.csv'
    train_output_path='data/processed/train.npy'
    test_output_path='data/processed/test.npy'
    transformer_path= 'Models/transformer.pkl'

    # call the transformation class
    transformer=DataTransformation(
        data_path=data_path,
        train_output_path=train_output_path,
        test_output_path=test_output_path,
        transformer_path=transformer_path
    )

    transformer.process()