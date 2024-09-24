# from src.utils import load_model
import pickle as pkl
import numpy as np
import logging
from sklearn.metrics import (r2_score,mean_absolute_error,mean_squared_error)
from sklearn.model_selection import cross_val_score
import pandas as pd
import json
import mlflow 
import mlflow.pyfunc
from mlflow import MlflowClient
import os
import dagshub
from dotenv import load_dotenv
load_dotenv()
dagshub_token = os.getenv('DAGSHUB_TOKEN')

if dagshub_token:
    os.environ['MLFlow_TRACKING_USERNAME']=dagshub_token
    os.environ['MLFlow_TRACKING_PASSWORD']=dagshub_token
    # Set up the MLflow tracking URI with authentication using the token
    mlflow.set_tracking_uri(f'https://{dagshub_token}:@dagshub.com/Sami606713/health_impact_analysis.mlflow')

    print("DagsHub login successful!")
else:
    print("DagsHub token not found. Please set the DAGSHUB_TOKEN environment variable.")
logging.basicConfig(level=logging.INFO)



class ModelEvulation:
    def __init__(self,test_data_path,model_name,report_path,alais):
        self.test_data=np.load(test_data_path)
        self.model_name = model_name
        self.report_path=report_path
        self.alais=alais
    
    def evulate_model(self):
        """
        This Fun is responsible for testing the model.
        input: Test Data
        output: Model Report
        """
        x_test=self.test_data[:,:-1]
        y_test=self.test_data[:,-1]
        
        # load the model
        model=self.load_model()
        y_pred=model.predict(x_test)

        test_score=cross_val_score(model,x_test,y_pred,cv=5,scoring="r2").mean()
        
        report=self.get_report(actual=y_test,y_pred=y_pred,x_test=x_test,test_score=test_score)

        if report['Adjusted R2 Score']*100 >= 90:
            print(f"{report['Adjusted R2 Score']} is > 90")
            self.prompote_model()

        logging.info(f"Report:\n{pd.DataFrame(report,index=[0])}")
        logging.info(f"Saving Report {self.report_path}")
        self.save_report(report_path=self.report_path,report=report)


    def load_model(self):
        """
        This fun is responsible for loading the model.
        input: Modle Path
        output: Model
        1- load the latest staging model and test its performance if performance is >90.
        2- Get the model version and prompte the model in producion
        """
        try:
            logging.info("Loading Model...")
            model_uri = f"models:/{self.model_name}@dev" 
            model = mlflow.sklearn.load_model(model_uri) 
            logging.info(f"{self.model_name} Loaded")
            return model
        except FileNotFoundError as e:
            logging.error(str(e))
    
    def prompote_model(self):
        """
        This fun is responsible for prompte the model staging --> production.
        """
        try:
            client=MlflowClient()
            latest_version=client.get_model_version_by_alias(self.model_name, self.alais).version
            if not latest_version:
                logging.error(f"No versions available for model '{self.model_name}' in alais '{self.alais}'.")

            # Prompte  the model
            print(f"{self.model_name} loaded with version {latest_version}")
            client.set_registered_model_alias(self.model_name,
                                              "champion", 
                                              version=latest_version)
            print("Model Prompted successfully.....")
        except Exception as e:
            return str(e)

    def adjusted_r2(self,r2, n, p):
        """
        Function to calculate the adjusted R² score.
        r2: R² score of the model.
        n: Number of observations.
        p: Number of features.
        """
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def get_report(self,actual,y_pred,x_test,test_score):
        r2=r2_score(actual,y_pred)
        return {
            "Mean Squared Error":mean_squared_error(actual,y_pred),
            "Mean Absolute Error": mean_absolute_error(actual,y_pred),
            "R2 Score":r2,
            "Adjusted R2 Score":self.adjusted_r2(r2=r2,n=x_test.shape[0],p=x_test.shape[1]),
            "Test Score":test_score
        }

    def save_report(self,report_path,report):
        try:
            with open(report_path,"w")as f:
                json.dump(obj=report,fp=f,indent=4)
        except Exception as e:
            logging.error(str(e))
            raise


if __name__=="__main__":
    evulation=ModelEvulation(test_data_path='data/processed/test.npy',model_name="final_model",
                             report_path='reports/model_report.json',alais="dev")
    evulation.evulate_model()


# # ========================================================
# import mlflow 
# import mlflow.pyfunc
# from mlflow import MlflowClient
# import os
# import dagshub
# from dotenv import load_dotenv
# load_dotenv()
# dagshub_token = os.getenv('DAGSHUB_TOKEN')

# if dagshub_token:
#     os.environ['MLFlow_TRACKING_USERNAME']=dagshub_token
#     os.environ['MLFlow_TRACKING_PASSWORD']=dagshub_token
#     # Set up the MLflow tracking URI with authentication using the token
#     mlflow.set_tracking_uri(f'https://{dagshub_token}:@dagshub.com/Sami606713/health_impact_analysis.mlflow')

#     print("DagsHub login successful!")
# else:
#     print("DagsHub token not found. Please set the DAGSHUB_TOKEN environment variable.")

# from mlflow import MlflowClient


# client = MlflowClient()
# model_name = "final_model"
# alias = "dev"  # or any other alias you are interested in

# # Get the model version by alias
# model_version = client.get_model_version_by_alias(model_name, alias)

# print(f"Latest version of the model with alias {alias}: {model_version.version}")
