"""
Using the california housing prices dataset 
"""
#imports 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



#FILE PATH
FILE_PATH_1 = "/home/abhi/Datasets/housing.csv"
FILE_PATH_2 = "/home/ubuntu/s3/housing.csv"  #for the ec2 machine 


class Data():
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    

    def make_data(self,FILE_PATH):
        """
        The function to make thge data set
        """
        self.df = pd.read_csv(FILE_PATH)

        self.X = self.df.drop(["median_house_value"],axis = 1)

        self.X = pd.get_dummies(self.X, columns=["ocean_proximity"], dtype= int)


        self.y = self.df["median_house_value"]

    
    def split_data(self):
        """
        split the dataset in values
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.33, random_state=42)

    def data_metrics(self):
        """
        Find the data correlations in the dataset
        """
        corr = self.X.corr()

        print(corr)
        


if __name__ == "__main__":
    data = Data()
    data.make_data(FILE_PATH_2)
    data.split_data()
    data.data_metrics()
    #print(data)
    #print(data.y_train)
