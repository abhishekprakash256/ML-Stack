"""
Make the dataset read and write in pyspark 
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer


# Initialize a SparkSession
spark = SparkSession.builder.appName("example").getOrCreate()

# Read data from a source and create a DataFrame (replace 'source_path' with your actual data source)
df_pyspark = spark.read.csv('test-sheet.csv', header=True, inferSchema=True)

#show the dataframe 

print("Show full data frame ")
print(df_pyspark.show())


print("The dropped data frame")
# Now you can perform DataFrame operations on df_pyspark
print(df_pyspark.drop("Name").show())


print("Null dropper")
print(df_pyspark.na.drop().show())

#drop as per threshold 
print("drop the threshold")
print(df_pyspark.na.drop(how="any",thresh = 2).show())


#handling the missing values in the data frame 
imputer = Imputer(
    inputCols=["Age","Experience","Salary"],  # List of columns with missing values
    outputCols=["Age","Experience","Salary"],  # New columns to store imputed values
    strategy="mean"  # Imputation strategy (other options: "median", "mode", "value")
)

print(imputer.fit(df_pyspark).transform(df_pyspark).show())

transformed_df = imputer.fit(df_pyspark).transform(df_pyspark)


#print("---------The original data that is given--------")
print(df_pyspark.show())

#print("---------The transformed data--------")
print(transformed_df.show())


#makng the fitering for the dataset 

print("-----------Running the salary filter--------------")
print(df_pyspark.filter("Salary<=25").show())

print("----------print the Name and age column-------------")
print(df_pyspark.filter("Salary<=25").select("Name","age").show())


print("print the schema of the dataset")
print(df_pyspark.printSchema())


#the groupby function 
print("the group by function of the scheme")
print(df_pyspark.groupBy("Age").sum().show())