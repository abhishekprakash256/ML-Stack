"""
The skklearn pipleine for the data
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
import pandas as pd



FILE_PATH = "/home/ubuntu/s3/drawndata1.csv"

df = pd.read_csv(FILE_PATH)

print(df["z"].unique())