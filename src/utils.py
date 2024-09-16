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