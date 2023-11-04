"""
Using the california housing prices dataset 
"""
#imports 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split





#FILE PATH
FILE_PATH = "/home/abhi/Datasets/housing.csv"


class Data():
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    

    def make_data(self):
        """
        The function to make thge data set
        """

        self.df = pd.read_csv(FILE_PATH)





if __name__ == "__main__":
    data = Data()
    data.make_data()