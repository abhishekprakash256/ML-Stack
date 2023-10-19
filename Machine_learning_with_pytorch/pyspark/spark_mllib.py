"""
using the spark mllib 
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer



# Initialize a SparkSession
spark = SparkSession.builder.appName("mlib").getOrCreate()

# Read data from a source and create a DataFrame (replace 'source_path' with your actual data source)
df_pyspark = spark.read.csv('test-sheet.csv', header=True, inferSchema=True)

#show the dataframe 

print("Show full data frame ")
print(df_pyspark.show())
