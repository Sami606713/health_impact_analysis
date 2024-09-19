from sklearn.utils import all_estimators
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import logging
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

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
        regressors = all_estimators(type_filter='regressor')
        counter=0
        for name,model_class in regressors:
            if counter==4:
                break
            else:
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
                        self.result["Adjusted"].append(self.adjusted_r2(r2=r2,n=self.x_test.shape[0],p=self.x_test.shape[1]))
                        self.result['Train_Score'].append(train_cross_val)
                        self.result['Test_Score'].append(test_cross_val)
                        self.result['Time'].append(time.time()-curr_time)
                        counter+=1
                    else:
                        logging.warning(f"Cross-validation for {name} produced NaN values, skipping this model.")
        
                except Exception as e:
                    continue
        logging.info('Taining Complete......')
        results=pd.DataFrame(self.result).sort_values(by=['Adjusted','Train_Score',"Test_Score"],ascending=[False,False,False])
        
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
        
if __name__=="__main__":
    # Sample Data (Use real data here)
    x_train = np.random.rand(100, 4)  # 100 samples, 4 features
    y_train = np.random.rand(100)
    x_test = np.random.rand(20, 4)    # 20 samples, 4 features
    y_test = np.random.rand(20)
    
    reg=LazyRegressor(x_train, y_train, x_test, y_test)
    # reg.train_all()
    best_model=reg.get_best_model()
    print("Prediction: ",best_model.predict(x_test))