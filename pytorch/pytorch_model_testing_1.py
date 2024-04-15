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
        self.fc = nn.Linear(10, 1)  # Fully connected layer with input size 10 and output size 1

    def forward(self, x):
        return self.fc(x)


model = SimpleNet()

# Define a loss function
criterion = nn.MSELoss()

# Define an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate some random input data
input_data = torch.randn(32, 10)  # 32 samples with 10 features each

# Forward pass
output = model(input_data)

# Compute the loss
target = torch.randn(32, 1)  # Target values for the regression task
loss = criterion(output, target)

# Backward pass (compute gradients)
optimizer.zero_grad()
loss.backward()

# Update model parameters
optimizer.step()

print(model)


