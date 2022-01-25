#make the ANN in the pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_features=4, h1 = 8, h2=9, out_features=3):
        #making the layers
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)

    def forward(self,x):
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= self.out(x)

        return x

torch.manual_seed(32)
#made a instance of model
model = Model()
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/iris.csv')
#print(df.tail())
X = df.drop("target", axis=1)
y= df['target']
X = X.values
y= y.values
#print(y)
#convert them to the numpy values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=0.2, random_state = 33)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#print(model.parameters())
#make the epochs
epochs = 100
losses = []
for i in range(epochs):
    #a forward pass
    y_pred = model.forward(X_train)
    #calculation of the loss
    loss = criterion(y_pred,y_train)
    losses.append(loss)
    if i%10 == 0:
        print(f'Epoch {i} and loss is: {loss}')
    #backward propogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs),losses)
plt.ylabel('LOSS')
plt.xlabel('EPOCH')
#plt.show() uncomment to plot

#to test the data in the test set
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval,y_test)
    print(loss)
correct = 0
with torch.no_grad():
    for i,data  in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i+1}.) {str(y_val)} {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct+=1
print(f'we got {correct} correct!')

#save the model to a file and pass in the directory in it
torch.save(model.state_dict(),'data/my_iris_model.pt')
