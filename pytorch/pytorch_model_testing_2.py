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
        self.fc1 = nn.Linear(1, 5)  # Fully connected layer with input size 10 and output size 1
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, 1) 

    def forward(self, x):
        x =  self.fc1(x)
        x =  self.fc2(x)
        x =  self.fc3(x)

        return x
    

model = SimpleNet()

# Make the dummy data
# Here, X has a shape of (batch_size, 1) to represent batched data
X = torch.randn(100, 1)  # Example with batch size 32
Y = X * 2.2 + 1.7

# Define a loss function
criterion = nn.MSELoss()

# Define an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# The epoch 
epochs = 200 

for epoch in range(epochs):
    # Forward pass
    output = model(X)

    # Compute the loss
    loss = criterion(output, Y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every few epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

print("The training is done")

# Test the model with a single data point
x_test_one = torch.randn(1, 1)  # Generate a single data point with 1 feature
output = model(x_test_one)
print("Input:", x_test_one.item())
print("Predicted output:", output.item())
print("Actual Value",x_test_one*2.2 + 1.7 )



