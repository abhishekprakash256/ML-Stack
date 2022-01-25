#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
#make the train trasnsform
train_transform = transforms.Compose([
transforms.RandomRotation(10),
transforms.RandomHorizontalFlip(),
transforms.Resize(224),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

])

test_transform = transforms.Compose([
transforms.Resize(224),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

#defining the path
root = './data/CATS_DOGS/'
#traindata
train_data = datasets.ImageFolder('./data/CATS_DOGS/train',transform = train_transform)
test_data = datasets.ImageFolder('./data/CATS_DOGS/test',transform = test_transform)
torch.manual_seed(42)

#making the data loader
train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size= 10)
#class names
class_names = train_data.classes
#printing the class names
#print(class_names)
#checking the length
#print(len(train_loader))
#lets grad a batch in the dataset
for images,labels in train_loader:
    break
#print(images.shape)
#visulalization of the batch
#im = make_grid(images,nrow=5)
#inv_normalize = transforms.Normalize(
#mean = [-0.485/0.229,-0.456/0.224,-0.406/0.225],
#std = [1/0.229,1/0.224,1/0.225]
#)
#im_inv = inv_normalize(im)
#plt.figure(figsize=(12,4))
#plt.imshow(np.transpose(im_inv.numpy(),(1,2,0)))
#plt.show()
#make the model
class ConvulationalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)#3 channels, 6 filers,
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(54*54*16,120) #120 number of neurons
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2) #last layer
    def forward(self,X):
        X = F.relu(self.conv1(X)) #first relu from the conv layer
        X = F.max_pool2d(X,2,2) #max pooling layer
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        #now flatten the data for the linaer network
        X = X.view(-1,54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X,dim=1)
#make the loss function
torch.manual_seed(101)
model = ConvulationalNetwork()
criterion = nn.CrossEntropyLoss()#loss function
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) #optimization function
#print(model)
#trainng part of the algorithm
import time
start_time = time.time()

#total time calulation

epochs = 3
#limits on the number of batches
max_train_btch = 800 #each batch has 100 images
max_test_btch = 300 #each bacth has 10 images

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_corr = 0
    test_corr = 0

    for b,(X_train,y_train) in enumerate(train_loader):

        #Limit the number of the batches
        if b == max_train_btch:
            break
        b+=1

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        #tally the number of predications
        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted == y_train).sum()
        train_corr +=batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%200 ==0 :
            print(f'Epoch {i} LOSS: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(train_corr)

    with torch.no_grad():
        for b,(X_test,y_test) in enumerate(test_loader):
            if b ==max_test_btch:
                break
            y_val = model(X_test)
            predicted = torch.max(y_pred.data,1)[1]
            batch_corr = (predicted == y_test).sum()
            test_corr = test_corr + batch_corr

    loss = criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)

#save the model now
torch.save(model.state_dict(), 'my_new_model.pt')





total_time = time.time() - start_time
#priting the total time
print(f'Total time is {total_time/60} minutes')
