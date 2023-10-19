"""
take the tips dataset and play with it 
"""

import pandas as pd 
import torch as th 


#data file path
FILE_PATH = "./datasets/tips.csv"


class Data:

    def __init__(self):
        self.df = df
        self.X = X
        self.y = y


    def make_data(self):
        """
        get the data and split in the data in X and y 
        """
        self.df = pd.read_csv(FILE_PATH)





if __name__ == "__main__":
    data = Data()
    