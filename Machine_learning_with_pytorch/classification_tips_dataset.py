"""
take the tips dataset and play with it 
"""

import pandas as pd 
import torch as th
from torch import nn
from sklearn.model_selection import train_test_split



#data file path
FILE_PATH = "./datasets/tips.csv"


class Data:

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
        self.df = pd.get_dummies(self.df, columns=['day'])
        self.df = self.df.astype(float)
        self.df = self.df.rename(columns = {'day_Friday': 'Friday', 'day_Tuesday': 'Tuesday'})
        
        col = ["week","n_peop","bill","Friday","Tuesday"]

        self.X = self.df[col]
        self.y = self.df["tip"]

    def split_data(self):
        """
        split the data in test and train
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.33, random_state=42)
        
        
        self.X_train = th.tensor(self.X_train.values, dtype= th.float32)
        self.X_test = th.tensor(self.X_test.values)
        self.y_train = th.tensor(self.y_train.values)
        self.y_test = th.tensor(self.y_test.values)

        


class Liner_Model(nn.Module):
    """
    Make a linear model
    """

    def __init__(self):
        super(Liner_Model, self).__init__()
        self.linear = nn.Linear(5, 1)  #input dims =  5 , out put = 1
 
    def forward(self, x):
        x = self.linear(x)
        return x


def train_test():
    """
    The fumctipm fdpr yje    
    """ 


    #data handling 
    model = Liner_Model()
    data = Data()
    data.make_data()
    data.split_data()

    #train the model
    




if __name__ == "__main__":
    train_test()