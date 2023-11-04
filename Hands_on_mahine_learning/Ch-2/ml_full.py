"""
Using the california housing prices dataset 
"""

import pandas as pd
import numpy as np


#FILE PATH
FILE_PATH = "/home/abhi/Datasets/housing.csv"


df = pd.read_csv(FILE_PATH)


print(df.head())
print(df.info())