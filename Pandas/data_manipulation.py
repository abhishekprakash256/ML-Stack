"""
pandas to maipulate the file
"""
import pandas as pd 
import dask.dataframe as dd

FILE_PATH = "/home/ubuntu/s3/ihateabhi.csv"

df = dd.read_csv(FILE_PATH)


print(df.info())
#df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1')