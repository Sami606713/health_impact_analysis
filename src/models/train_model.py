import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.utils import save_model
import os
import logging
logging.basicConfig(level=logging.INFO)


class ModelTraining:
    def __init__(self,train_data_path,model_output_path):
        self.train_data_path=train_data_path
        self.model_output_path=model_output_path
        self.train_data=np.load(train_data_path)
    
    def saperate_feature_label(self,train_data):
        """
        This fun is responsible for saperating the feature and label from the dataset
        """
        try:
            logging.info("Saperating the feature and label")
            x_train=train_data[:,:-1]
            y_train=train_data[:,-1]

            logging.info(f"X_train: {x_train.shape}, y_train: {y_train.shape}")

            return x_train,y_train
        except Exception as e:
            logging.error(f"Error in saperating the feature and label: {e}")
            raise
    
    def train_model(self):
        """
        This fun is responsible for training the model
        input: x_train,y_train
        output: model
        """
        try:
            # get the feature and label
            x_train,y_train=self.saperate_feature_label(train_data=self.train_data)

            model=RandomForestRegressor()

            logging.info("Training the model")
            model.fit(x_train,y_train)
            logging.info(f"Model Training Completed")

            if self.model_output_path:
                logging.info(f"Saving model {self.model_output_path}")
                save_model(model=model,output_path=self.model_output_path)

        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise
        

if __name__=="__main__":
    # set the paths train_data_path,model_output_path
    train_data_path='data/processed/train.npy'
    model_output_path='Models/model.pkl'

    # call the transformation class
    trainer=ModelTraining(train_data_path=train_data_path,
                          model_output_path=model_output_path)
    trainer.train_model()
    