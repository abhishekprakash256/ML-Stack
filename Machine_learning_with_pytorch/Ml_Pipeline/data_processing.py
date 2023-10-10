"""
The file to load the data from the storage into memeory 
"""


#imports 
import torch as th


# Specify a file path for storing the tensor
X_FILE_PATH = "./data/x.pt"
Y_FILE_PATH = "./data/y.pt"



class data:

    def load_data(self):
        """
        The method to load the data 
        """

        x = th.load(x_file_path)
        y = th.load(y_file_path)