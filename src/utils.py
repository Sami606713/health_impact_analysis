import pickle as pkl
import os 
import logging

logging.basicConfig(level=logging.INFO)



def save_model(model,output_path):
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

def load_model(model_path):
    """
    This fun is responsible for loading the model.
    input: Modle Path
    output: Model
    """
    try:
        if os.path.exists(model_path):
            with open(model_path,"rb") as f:
                model=pkl.load(f)
                logging.info(f"Model Loaded Successfully from this path: {model_path}")
                return model
    except FileNotFoundError as e:
        logging.error(str(e))
        raise

# if __name__=="__main__":
#     load_model("Models/model.pkl")