"""
making a linear regresion from scratch
"""


import torch as th



#make the fundamental of linear regression 

weight = 0.7 
bias = 0.3

#create 
start = 0
end = 1 
step = 0.02

X = th.arange(start,end,step).unsqueeze(dim=1)

#make the regression

Y = weight*X + bias



#make the test and train split 

train_split = int(0.8*len(X))

X_train , y_train = X[:train_split],Y[:train_split]
X_test , y_test = X[train_split:],Y[train_split:]


print(len(X_train),len(X_test))