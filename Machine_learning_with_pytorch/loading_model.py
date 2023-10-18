"""
load the model and work on the model
"""

#imports 
import torch as th 
from pathlib import Path
from linear_regression import Linear_regression_model




#--model save information ----------------
model_path = Path("models")
model_name = "linear_regression_01.pth"
model_save_path = model_path / model_name



#make model 

load_model_0 = Linear_regression_model()


print(f"The parameters before loading the model: {load_model_0.state_dict()}")

#load the model 

load_model_0.load_state_dict(th.load(f=model_save_path))


print(f"The parameters after loading the model: {load_model_0.state_dict()}")