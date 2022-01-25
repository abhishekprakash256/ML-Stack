#imports
import torch 
from torch import nn
from torch import optim

from torchvision import datasets, transforms
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()


model = nn.Sequential(
    #make the layers 
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.Softmax(dim =1)

)

#caclulation of the loss

criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))

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





