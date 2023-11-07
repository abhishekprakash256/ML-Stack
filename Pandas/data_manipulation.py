"""
pandas to maipulate the file
"""
import pandas as pd 
import dask.dataframe as dd

FILE_PATH = "/home/ubuntu/s3/ihateabhi.csv"

df = dd.read_table(FILE_PATH,encoding='ISO-8859-1')


print(df.head(0))
#df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1')