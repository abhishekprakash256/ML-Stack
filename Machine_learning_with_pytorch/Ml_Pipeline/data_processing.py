"""
The file to load the data from the storage into memeory 
"""


#imports 
import torch as th
from sklearn.model_selection import train_test_split

# Specify a file path for storing the tensor
X_FILE_PATH = "./data/x.pt"
Y_FILE_PATH = "./data/y.pt"



class data:

    def load_data(self):
        """
        The method to load the data 
        """

        x = th.load(X_FILE_PATH)
        y = th.load(Y_FILE_PATH)

        

