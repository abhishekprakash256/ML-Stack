#imports 
import torch 
import numpy as np

#generate some dummy data 
np.random.seed(42)  #make a seed for random function

x = np.random.rand(100,1)

#our equation is y = a +bx + c

y = 1 + 2*x + .1 * np.random.randn(100,1)  #equation is initilaized

idx = np.arange(100)

np.random.shuffle(idx)

#using for the train 

train_idx = idx[:80]

#using for the test idx

test_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]  # from indexing the values are randomly used 
x_val, y_val = x[test_idx], y[test_idx]

#make the model predictions 

np.random.seed(42)  # seed initilaization

a = np.random.randn(1)
b = np.random.randn(1)


#setting up the learning rate 
lr = 1e-1 #for the power of the e it shpuld be close 

epochs = 1000  #making the epochs 

#trainng the model for the numpy 

for epoch in range(epochs):
    
    #toy model 
    pred = a + b*x_train  #make the prediction

    #error calculation in the model

    error = (y_train - pred)

    loss = (error ** 2).mean() #compute the MSE error 

    #print(loss)

    #compute the gradients 

    a_grad = -2* error.mean()

    b_grad = -2* (x_train * error).mean()

    #update the paramaters
    a = a - lr* a_grad
    b = b - lr* b_grad


print(a,b)

# Sanity Check: do we get the same results as our gradient descent?
from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])