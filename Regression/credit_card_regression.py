"""
train the model on the credit card dataset
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
import dask.dataframe as dd


#STATIC 
FILE_PATH = "/home/ubuntu/creditcard.csv"


df = pd.read_csv(FILE_PATH, chunksize=10000)

print(df.columns )