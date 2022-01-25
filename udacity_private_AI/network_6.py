#imports from the pytorch 

import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()



class Network(nn.Module):
    def __init__(self):
        super().__init__()

    #make the network 

        self.hidden_1 = nn.Linear(784,128) #input layer

        self.hidden_2 = nn.Linear(128,64)

        self.output = nn.Linear(64,10) #out layer 

    def forward(x):
        x = F.relu(self.hidden(x))  #the hidden layer
        x = F.relu(self.hidden_2(x)) 
        x = F.softmax(self.output(x), dim = 1) #the output 



model = Network()

criterion = nn.NLLLoss()

#flatten the images

images = images.view(images.shape[0],-1)

logits = model(images)

loss = criterion(logits, labels)

loss.backward()

#using the optimizer 

optimizer = optim.SGD(model.parameters(), lr=0.01) #optimizer and learning 

#training starts 
epochs = 5

for i in range(epochs):
    running_loss = 0
    for images,labels in trainloader:
        images = images.view(images.shape[0],-1) #flatten the data

        #make the gradient zero in loops 
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        running_loss +=loss.item()
    else:
         print(f"Training loss: {running_loss/len(trainloader)}")

