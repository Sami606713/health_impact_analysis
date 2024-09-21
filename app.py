from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import pickle as pkl
import os
import dagshub
dagshub.init(repo_owner='Sami606713', repo_name='health_impact_analysis', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Sami606713/health_impact_analysis.mlflow')


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


def load_model(model_name,model_version):
    """
    This fun is responsible for loading the model.
    input: Modle Path
    output: Model
    """
    try:
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
                # in 2-3 days set the version is production
            model_uri = f"models:/{model_name}/{model_version}"  # Load latest production version
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
        model=load_model("final_model",1)

        # # prediction
        response=model.predict(final_data)

        # Ensure the response is in a serializable format (like a list)
        serializable_response = response.tolist() if hasattr(response, 'tolist') else response

        return {"prediction": serializable_response}
    except Exception as e:
        return {"Error": str(e)}
