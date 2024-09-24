from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
from dotenv import load_dotenv
import pandas as pd
import mlflow
import pickle as pkl
import dagshub
import uvicorn
import os
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


app = FastAPI()

class HealthPredictionInput(BaseModel):
    datetimeEpoch: float
    tempmax: float
    tempmin: float
    temp: float
    feelslikemax: float
    feelslikemin: float
    feelslike: float
    dew: float
    humidity:float
    snow: float
    snowdepth: float
    windgust: float
    solarradiation: float
    uvindex: float
    sunriseEpoch:float
    sunsetEpoch: float
    moonphase: float
    conditions: str
    description: str
    icon: str
    source: str
    City:str
    Temp_Range:float
    Heat_Index:float
    Severity_Score: float
    Day_of_Week: str
    Is_Weekend: bool
    sunset_hour:int
    sunrise_hour:int
    sunrise_minute:int


def load_processor(processor_path:str):
    """
    This fun is responsible for loading the processor.
    Input: input_path
    output: processor 
    """
    try:
        if os.path.exists(processor_path):
            with open (processor_path,"rb")as f:
                processor=pkl.load(f)
                return processor
        else:
            return f"{processor_path} does not found"
            
    except Exception as e:
        return str(e)


def load_model(model_name):
    """
    This fun is responsible for loading the model.
    input: Modle Path
    output: Model
    """
    try:
        # in 2-3 days set the version is production
        client=MlflowClient()
        latest_version=client.get_model_version_by_alias(model_name, "champion").version
        if not latest_version:
            return (f"No versions available for model '{model_name}' with alais champion.")
        else:
            model_uri = f"models:/{model_name}@champion" 
            model = mlflow.sklearn.load_model(model_uri) 
        
        return model
    except FileNotFoundError as e:
        return str(e)



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(heatlth_data: HealthPredictionInput):
    try:
        # Convert the input data to a pandas DataFrame
        input_data = pd.DataFrame([heatlth_data.dict()],index=[0])
        processor = load_processor(processor_path='Models/transformer.pkl')

        # # transform the data
        final_data= processor.transform(input_data)
        # # load model
        model=load_model("final_model")

        # # prediction
        response=model.predict(final_data)

        # Ensure the response is in a serializable format (like a list)
        serializable_response = response.tolist() if hasattr(response, 'tolist') else response

        return {"prediction": serializable_response}
    except Exception as e:
        return {"Error": str(e)}


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)