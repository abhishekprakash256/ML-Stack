#imports
import torch

from torch import nn

from torchvision import datasets, transforms
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

#make the network with the init module

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #inputs to the tensor 
        self.hidden = nn.Linear(784,256)

        self.output = nn.Linear(256,10)

        #fucntions
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

    def forward(x):
        #passing the tensors to make the operations

        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x



model = Network()
print(model)




