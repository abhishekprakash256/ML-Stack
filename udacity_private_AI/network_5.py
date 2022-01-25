#imports from the pytorch 

import torch 
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #creating the layers 
        self.hidden = nn.Linear(784,256)

        #outputs are here 
        self.output = nn.Linear(256,10)

    def forward(x):
        x = F.sigmoid(self.hidden(x))  #the hidden layer
        x = F.softmax(self.output(x), dim = 1) #the output layer

        return x 

model = Network()

print(model)

    






    


