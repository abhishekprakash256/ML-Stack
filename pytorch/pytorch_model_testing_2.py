"""
make the pytorch dummy model and train on dummy data just on an array 
make a linear array 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim



#make the model 

# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(100, 5)  # Fully connected layer with input size 10 and output size 1
        self.fc2 = nn.Linear(5, 100) 

    def forward(self, x):
        x =  self.fc1(x)
        x =  self.fc2(x)

        return x
    

model = SimpleNet()

#make the dummy data
X = torch.randn(1, 100)
Y = X*2.2 + 1.7

#make the optmizer function 
# Define a loss function
criterion = nn.MSELoss()

# Define an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)


#the epoch 
epoch = 200 

for i in range(epoch):

    output = model(X)

    loss = criterion(output, Y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

print("The training is done")

x_test_one = torch.randn(1, 100)

print(x_test_one)

print(model(x_test_one))