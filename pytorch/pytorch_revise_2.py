#imports 
import torch 
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import numpy as np


#generattion of the data 

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


device = 'cuda' if torch.cuda.is_available() else 'cpu'  #cuda for the GPU


#conversion to the torch tesnsor

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

#the model making from the pytorch

#initilaize the random varibales for a and b

#make the seed for the random
torch.manual_seed(42)


a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

#making the dummy model

lr = 1e-1

epochs = 100

for epoch in range(epochs):
    ypred = a + b*x_train_tensor # make the prediction from the values
    error = y_train_tensor - ypred

    loss = (error**2).mean() #calculation of the loss

    loss.backward() # loss calculation
    print(a.grad)
    print(b.grad)

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    a.grad.zero_()
    a.grad.zero_()

print(a,b)










    

