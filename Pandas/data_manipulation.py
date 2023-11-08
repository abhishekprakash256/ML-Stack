"""
pandas to maipulate the file
"""
import pandas as pd 
import dask.dataframe as dd

FILE_PATH_1 = "/home/ubuntu/s3/ihateabhi.csv"
FILE_PATH_2 = "/home/abhi/Datasets/ihateabhi.csv"

df = pd.read_csv(FILE_PATH_2,encoding='ISO-8859-1')

#pprint the info of the data

print(df.info())

print(df["BorrState"].unique())

print(df.loc[df["BorrState"] == "CA"])

## add two condition and connect by logic

print(df.loc[(df["BorrState"]== "CA") & (df["TermInMonths"] <= 120)])

#find the value of the state isin BorrowState 
new_frame = (df.loc[df["BorrState"].isin(['CA','IL','CO','OH','WA'])])

print(new_frame["BorrState"].unique())

#the first row 
print(df.loc[0])

#print the row with values and filter with columns

val = ["BorrState","TermInMonths"]
indics = [0,1,10,100]

print(df.loc[indics,val])


