#all the imports
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#conversion of the mnist data to tensors
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data",train=True, download=True,transform=transform)
test_data = datasets.MNIST(root="./data",train=False, download=True,transform=transform)
#print(train_data)
#print(test_data)
#print(train_data[0])
image,label = train_data[0]
#print(image)
#print(image.shape)
#print(label)
plt.imshow(image.reshape(28,28),cmap='gist_yarg')# change the default color scheme to something else
#plt.show()
#seeting the manual_seed
torch.manual_seed(101)
train_loader = DataLoader(train_data,batch_size=100,shuffle=True)
test_data = DataLoader(test_data,batch_size=500, shuffle=False)
#import the torchvision
from torchvision.utils import make_grid
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
#First batch
for images,labels in train_loader:
    break
#print(images.shape,labels.shape)
#make the network, 3 layers and 120 and 84 neurons
class MultilayerPerceptron(nn.Module):
    def __init__(self,in_sz=784,out_sz=10,layers=[120,84]):#the number of the neurons
        super().__init__()
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],out_sz)

    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X,dim=1)
        
torch.manual_seed(101)
#the model is initilalized
model = MultilayerPerceptron()
#print(model)
#print the features in the models
#for param in model.parameters():
    #print(param.numel())
#making the loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
#makeing the conversion of the shape now
#use view to make dmimension change
images.shape
#print(images.view(100,-1))

#to check for the time stamp
import time
start_time = time.time()
#make the epoch and do the iteration and also the loss checking
#training
epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    for b, (X_train,y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train.view(100,-1))
        loss = criterion(y_pred,y_train)

        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b%2 ==0:
            acc = trn_corr.item()*100/(100*b)
            print(f'Epoch {i} batch{b} loss: {loss.item()} accuracy:{acc}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    with torch.no_grad():


        for b,(X_test,y_test) in enumerate(test_data):
            y_val = model(X_test.view(500,-1))
            predicted = torch.max(y_val.data,1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

total_time = time.time() - start_time
print(f'Duration: {total_time/60} mins')
