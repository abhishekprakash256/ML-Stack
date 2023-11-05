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

        self.X = self.df.drop(["median_house_value"],axis = 1)

        self.y = self.df["median_house_value"]

    
    def split_data(self):
        """
        split the dataset in values
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.33, random_state=42)
        



if __name__ == "__main__":
    data = Data()
    data.make_data()
    data.split_data()
    print(data.X_train)
    print(data.y_train)