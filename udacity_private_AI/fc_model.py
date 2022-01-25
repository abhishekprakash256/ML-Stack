import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
#making the model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #the layers 
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.out = nn.Linear(64,10)
        #making a dropout solution
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
        #flatten the tensor 
        x = x.view(x.shape[0],-1)

        x = self.dropout(F.relu(self.fc1(x)))  #the hidden layer
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))


        x = F.softmax(self.out(x), dim = 1) #the output 

        return x

