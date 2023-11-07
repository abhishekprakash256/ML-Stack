"""
pandas to maipulate the file
"""
import pandas as pd 


FILE_PATH = "/home/ubuntu/s3/creditcard.csv"

df = pd.read_csv(FILE_PATH)

#print the column of dataset
print(df.Amount)

#print the Amount columns
print(df["Amount"])

#print the value in value in the column 
print(df["Amount"][0])

#using the iloc 
print(df.iloc[0])


#using the loc 
print(df.loc[0])


#using the column as target
print(df.loc[0,'Amount'])

#set the index in the 
print(df.set_index("Amount"))