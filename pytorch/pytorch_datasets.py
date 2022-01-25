#imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/iris.csv')
#print(df.describe)
#print(df.head())
#import the train test split
from sklearn.model_selection import train_test_split
features = df.drop('target', axis = 1).values
label = df['target'].values
#using the pytorch library
from torch.utils.data import TensorDataset , DataLoader
#dividing the dataset
data = df.drop('target', axis =1).values
labels = df['target'].values
#
iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
#to make the random data loader
iris_loader = DataLoader(iris, batch_size=50,shuffle= True)
#print(type(iris))
#print(len(iris))
#for i in iris:
#    print(i)
#make the batch it shuffles and make the random data as well
for i_batch in (iris_loader):
    print(i_batch)
