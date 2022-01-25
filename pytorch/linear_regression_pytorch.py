import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

#make a tensor with linear space as 50 elemets
X = torch.linspace(1,50,50).reshape(-1,1)
#seting the seed to get the same random  values
torch.manual_seed(71)
#make the error and bias
e = torch.randint(-8,9,(50,1),dtype=torch.float)
#print(e)
#to make the line and add the line as e
y = 2*X +1 +e

#plt.show() to see uncomment it
#print(X)
#print(y)
#print(y.shape)
#now we set the seed again
torch.manual_seed(59)

#make the linear model
#call from the nn
#we have one feture and one output
model = nn.Linear(in_features=1 , out_features=1)
#print the model weight
#print(model.weight)
#print the model bias
#print(model.bias)
#make the class for the linaer regression
'''inheritaed the class Model from the Module and made a layer linear and another forward method with y_pred as return '''
class Model(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
#as per this model that is inherired above
torch.manual_seed(59)
model = Model(1,1)
#print(model.linear.weight)
#print(model.linear.bias)

#printing the weight and linear bias from the Model by inheriting the model class
for name,param in model.named_parameters():
    print(name,'\t', param.item())
#getting the value from the tensor when x value is 2.0
x = torch.tensor([2.0])
#print(model.forward(x))
#lets make the tensor and do the first prediction with the Model
x1 = torch.linspace(0,50.0, 50)
#print(x1)
w1 = 0.1059
b1 = 0.9637

y1 = w1*x1 + b1
#print(y1)
#make the pyplot
plt.scatter(X.numpy(), y.numpy())
#plt.plot(x1,y1,'r')
#plt.show()
#now to optimize make the loss function
#for the optimization we use the mean square function
criterion = nn.MSELoss()
#using the stocahastioc gradidient descent, the paramaers are inherited from the model
#provide the learning rate
optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
#now we make the epoch
epochs = 50
losses = []
#to make the loss function and get all the losses
for i in range(50):
    i=+1
    #predting the forward pass
    y_pred = model.forward(X)
    #calulate the loss
    loss= criterion(y_pred,y)
    #record the loss
    losses.append(loss)

    print(f"epoch {i} loss: {loss.item()} weight: {model.linear.weight.item()} bias : {model.linear.bias.item()}")
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
#pltiing the losses as per the Model
x = np.linspace(0.0, 50.0,50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()

predicted_y = current_weight*x + current_bias
#print(predicted_y)
plt.plot(x,predicted_y,'r')
plt.show()
