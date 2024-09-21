import numpy as np
# from src.models.find_best_model import LazyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import logging
import time
import yaml
import warnings
import dagshub
dagshub.init(repo_owner='Sami606713', repo_name='health_impact_analysis', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Sami606713/health_impact_analysis.mlflow')
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
from src.utils import save_model
import os
import logging
logging.basicConfig(level=logging.INFO)


# ==================================Lazy Regressor Temporary=====================================#
class LazyRegressor:
    """
    This Class is responsible for training all the models.
    After Training return the best model

    input: x_train,x_test,y_train,y_test
    output: Best Model
    """

    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.models={}
        self.result={
            "name":[],
            "MAE":[],
            "MSE":[],
            "R2_Score":[],
            "Adjusted":[],
            "Train_Score":[],
            "Test_Score":[],
            "Time":[]
        }

    def adjusted_r2(self,r2, n, p):
        """
        Function to calculate the adjusted R² score.
        r2: R² score of the model.
        n: Number of observations.
        p: Number of features.
        """
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def train_all(self):
        """
        Function to to train all the models
        """
        logging.info("Traing Multiple Models Start......")
        mlflow.set_experiment("Multiple Models Tracking")
        regressors = all_estimators(type_filter='regressor')
        counter=0
        for name,model_class in regressors:
            if counter >= 10:  # Stop the loop if counter reaches 10
                logging.info("Counter reached 10, stopping the process.")
                break
            with mlflow.start_run(run_name=name):
                try:
                    curr_time=time.time()
                    model=model_class()
                    model.fit(self.x_train,self.y_train)
                    logging.info(f"Trainig====> {name}")
                    self.models[name]=model

                    y_pred=model.predict(self.x_test)
                    
                    mse=mean_squared_error(self.y_test,y_pred)
                    mae=mean_absolute_error(self.y_test,y_pred)
                    r2=r2_score(self.y_test,y_pred)
                    logging.info("Train Score Calculeted.....>")
                    train_cross_val=cross_val_score(model,self.x_train,self.y_train,cv=5,scoring="r2").mean()
                    test_cross_val=cross_val_score(model,self.x_test,self.y_test,cv=5,scoring="r2").mean()

                    if not (np.isnan(train_cross_val) or np.isnan(test_cross_val)):

                        self.result['name'].append(name)
                        self.result['MAE'].append(mae)
                        self.result['MSE'].append(mse)
                        self.result['R2_Score'].append(r2)
                        adjusted_r2=self.adjusted_r2(r2=r2,n=self.x_test.shape[0],p=self.x_test.shape[1])
                        self.result["Adjusted"].append(adjusted_r2)
                        self.result['Train_Score'].append(train_cross_val)
                        self.result['Test_Score'].append(test_cross_val)
                        self.result['Time'].append(time.time()-curr_time)

                        # Tracking model parameter and model metric
                        mlflow.log_metric("MAE", mae)
                        mlflow.log_metric("MSE", mse)
                        mlflow.log_metric("R2 Score", r2)
                        mlflow.log_metric("Adjusted R2", adjusted_r2)
                        mlflow.log_metric("Train_Score", train_cross_val)
                        mlflow.log_metric("Test_Score", test_cross_val)
                        mlflow.sklearn.log_model(model,name)
                        if hasattr(model, 'get_params'):
                            mlflow.log_param(name+"Params", model.get_params())
                        # Increment the counter for each successful run
                        counter += 1
                        #====================Model Tracking End =============================#

                    else:
                        logging.warning(f"Cross-validation for {name} produced NaN values, skipping this model.")
        
                except Exception as e:
                    continue
        logging.info('Taining Complete......')
        results=pd.DataFrame(self.result).sort_values(by=['Adjusted','Train_Score',"Test_Score"],ascending=[False,False,False])
        # Optionally save results as an artifact
        results.to_csv("reports/results.csv", index=False)
        mlflow.log_artifact("reports/results.csv")
        return results

    def get_best_model(self):
        """
        Function to return the best model
        """
        logging.info("Getting the best model")
        if len(self.models.keys())==0:
            results=self.train_all()
            logging.info(results)
        else:
            results=pd.DataFrame(self.result).sort_values(by=['Adjusted','Train_Score',"Test_Score"],ascending=[False,False,False])
            logging.info(results)
            
        best_model_name = results.iloc[0]['name']  # Best model based on Adjusted R²
        logging.info(f'Best Model: {best_model_name}')
        return self.models[best_model_name]
# ==================================Lazy Regressor Temporary=====================================#

# =================================Model Training=====================================#
class ModelTraining:
    def __init__(self,train_data_path,test_data_path,model_output_path):
        self.model_output_path=model_output_path
        self.train_data=np.load(train_data_path)
        self.test_data=np.load(test_data_path)
    
    def saperate_feature_label(self,train_data,test_data):
        """
        This fun is responsible for saperating the feature and label from the dataset
        """
        try:
            logging.info("Saperating the feature and label")
            x_train=train_data[:,:-1]
            y_train=train_data[:,-1]

            x_test=test_data[:,:-1]
            y_test=test_data[:,-1]
            logging.info(f"X_train: {x_train.shape}, y_train: {y_train.shape}")

            return x_train,y_train,x_test,y_test
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
            x_train,y_train,x_test,y_test=self.saperate_feature_label(train_data=self.train_data,
                                                                      test_data=self.test_data)

            # Training all the models
            reg=LazyRegressor(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
            reg.train_all()
            # Get Best model from the trained models
            best_model=reg.get_best_model()
            
            # Train the best model again and track them using mlflow
            with mlflow.start_run(run_name="final_model",nested=True):
                best_model.fit(x_train,y_train)

                y_pred=best_model.predict(x_test)
                mae=mean_absolute_error(y_test,y_pred)
                mse=mean_squared_error(y_test,y_pred)
                r2=r2_score(y_test,y_pred)

                # log them
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("R2 Score", r2)
                mlflow.sklearn.log_model(best_model,"Final_Model")
                if hasattr(best_model, 'get_params'):
                    mlflow.log_param("Final_model_params", best_model.get_params())
                    with open('Config/config.yml', 'w') as file:
                        yaml.dump(best_model.get_params(), file, default_flow_style=False)

                # Register the final model in mlflow registry
                # Register the model in the Model Registry
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/final_model"
                model_version = mlflow.register_model(model_uri=model_uri, name="final_model")


            if self.model_output_path:
                logging.info(f"Saving model {self.model_output_path}")
                save_model(model=best_model,output_path=self.model_output_path)

        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise
        
        

if __name__=="__main__":
    # Set the paths training data, testing data and model_output_path
    train_data_path='data/processed/train.npy'
    test_data_path="data/processed/test.npy"
    model_output_path='Models/model.pkl'

    # call the transformation class
    trainer=ModelTraining(train_data_path=train_data_path,test_data_path=test_data_path,
                          model_output_path=model_output_path)
    trainer.train_model()
    