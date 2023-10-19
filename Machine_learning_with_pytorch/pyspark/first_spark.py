"""
using  pyspark for data manipulation 
"""
from pyspark.sql import SparkSession
import pandas as pd




spark = SparkSession.builder.appName("Practise").getOrCreate()

#print(spark)

df_pyspark = spark.read.option('header','true').csv('test-sheet.csv')


print(df_pyspark.columns)

print(df_pyspark.dtypes)

print(df_pyspark.describe())

print(df_pyspark.describe().show())

print(df_pyspark.head())


#multiple data frame can be selected
print(df_pyspark.select("Name"))



