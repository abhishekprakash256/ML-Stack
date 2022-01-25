#the torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms
from torchvision.utils import make_grid

#other imports
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#all the import are done
#transformation of the data

transform = transforms.ToTensor()
#this is the train data
train_data = datasets.CIFAR10(root= "./data", train = True,download = True, transform = transform)
#the test data
test_data = datasets.CIFAR10(root="./data",train=False,download= True,transform=transform)
#verify and see the data number
#print(train_data)
#print(test_data)
#set the manual_seed
torch.manual_seed(101)
#make the data loader
train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10, shuffle = True)
#make the class names
class_name = ['plane', 'car','bird','cat','deer','dog','frog','horse','ship','truck']
#checking one batch
for images,labels in train_loader:
    break
#print(labels)
#make the model
class ConvulationalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5,1) #input channels, filters, kernel size, stride size
        self.conv2 = nn.Conv2d(6,16,3,1)#it will recive beacuse it has 6 filters
        self.fc1 = nn.Linear(6*6*16,120) #cause in the pooling we are loosing the pixels so we do the, (((32-2)/2)-2)/2
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,6*6*16) #data is flatten
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim=1)

#initilalized the model
model = ConvulationalModel()
#print(model)
#priting the paramaters in the model
#for param in model.parameters():
#    print(param.numel())
#make the loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

#the tarining phase
import time
start_time = time.time()

#variables (trackers)
epochs = 10
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

        if b%1000 == 0:
            print(f"EPOCH: {i} BATCH: {b} LOSS: {loss.item()}")


    train_losses.append(loss)
    train_correct.append(trn_corr)
    #TEST

    with torch.no_grad():
        #print("check")
        for b,(X_test,y_test) in enumerate(test_loader):
            #print("check_2")
            y_val = model(X_test)

            predicted = torch.max(y_val.data,1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)


current_time = time.time()
total = current_time-start_time
print(total)
