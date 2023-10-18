"""
make the regression model 
"""
import pandas as pd 
import torch as th 



DATA_SET_PATH = "../datasets/Melbourne_housing_FULL.csv"



df = pd.read_csv(DATA_SET_PATH)

print(df.describe())