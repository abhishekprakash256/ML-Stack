#imports for the file 
import numpy as np

#creating the values 
num_labels = 10 # the label for one of the data 
num_teachers = 10 #as 10 hospitals 
num_exapmles = 10000 #size of the dataset 

predictions = (np.random.rand(num_teachers, num_exapmles) * num_labels).astype(int)

print(predictions.shape)
