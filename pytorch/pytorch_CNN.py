#imports for the file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

#supporing libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#load the dataset
transform = transforms.ToTensor()
#the train data
train_data = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
#the test data
test_data = datasets.MNIST(root="./data",train=False,download=True,transform=transform)
#data visulalization
#print(train_data)
#print(test_data)
#create a loader
train_loader = DataLoader(train_data,batch_size=10, shuffle = True)
test_loader = DataLoader(test_data,batch_size=10, shuffle = True)
#making the model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim=1)

#set the seed
torch.manual_seed(42)
#create the model instance
model = Network()
#print(model)
#for param in model.parameters():
#    print(param.numel())
#make the loss function
criterion = nn.CrossEntropyLoss()
#make the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

#the tarining phase
import time
start_time = time.time()

#variables (trackers)
epochs = 5
train_losses =[]
test_losses=[]
train_correct=[]
test_correct=[]

#for the loop
for i in range(epochs):
    trn_corr =0
    tst_corr = 0

    #train
    for b,(X_train,y_train) in enumerate(train_loader):
        b+=1

        y_pred = model(X_train)
        loss = criterion(y_pred,y_train)

        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr +=batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%600 == 0:
            print(f"EPOCH: {i} BATCH: {b} LOSS: {loss.item()}")


    train_losses.append(loss)
    train_correct.append(trn_corr)
    #TEST

    with torch.no_grad():
        for b,(X_test,y_test) in enumerate(test_loader):
            y_val = model(X_test)

            predicted = torch.max(y_val.data,1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)


current_time = time.time()
total = current_time-start_time
print(total)
