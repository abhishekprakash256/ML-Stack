"""
linear regression with  student dataset , preparing the dataset as one hot encoding 
"""

#imports 
import pandas as pd
import torch as th
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


#data file path
FILE_PATH = "../datasets/student/tips_full.csv"
EPOCHS = 1000


class Data_Prepration:
    """
    class to prepare the data as 
    
    """

    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


    def make_data(self):
        """
        get the data and split in the data in X and y 
        """
        self.df = pd.read_csv(FILE_PATH)
        #self.y = self.df["tip"].astype(float)

        print(self.df)

        #make the train set 
        #col = ["total_bill","tip","sex","smoker","day","time","size"]
        #self.X = self.df[col]
        #self.X = pd.get_dummies(self.X).astype(float)


    def split_data(self):
        """
        split the data in test and train
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.33, random_state=42)

        self.X_train = th.tensor(self.X_train.values,dtype= th.float32)
        self.X_test = th.tensor(self.X_test.values, dtype= th.float32)
        self.y_train = th.tensor(self.y_train.values,  dtype= th.float32)
        self.y_test = th.tensor(self.y_test.values,  dtype= th.float32)



if __name__ == "__main__":
    data = Data_Prepration()
    data.make_data()