"""
handling the credit card dataset 
"""

import pandas as pd
import torch as th
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel




#file path 
FILE_PATH = "/home/abhi/Datasets/creditcard.csv"


class Data():
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
        self.df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1')




if __name__ == "__main__":
    data = Data()
    data.make_data()

    print(data.df.info())