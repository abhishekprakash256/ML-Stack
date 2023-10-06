"""
load the model and use the parameters 
"""
#imports 
import torch as th
from pathlib import Path
from linear_regression import Linear_regression_model


#make the parameters 
model_path = Path("models")

#create the save model 
model_name = "linear_regression_01.pth"
model_save_path = model_path / model_name


class Model_loader():
    def __init__(self):
        self.linear_model = Linear_regression_model()



if __name__ == "__main__":
    model_0 = Model_loader()
    model_0.linear_model

    param = model_0.linear_model.state_dict()

    print(param)

    #load the parameters and check the model params 

    model_0.linear_model.load_state_dict(th.load(f=model_save_path))

    param2 = model_0.linear_model.state_dict()

    print(param2)



