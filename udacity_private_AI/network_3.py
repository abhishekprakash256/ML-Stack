import torch

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

#create the network 

def activation(x):
    return 1/(1+torch.exp(-x))

#flatten the data 
inputs = images.view(images.shape[0],-1)

#create the paramater

w1 = torch.randn(784,256)
b1 = torch.randn(256)

w2 = torch.randn(256,10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs,w1)+ b1)

out = torch.mm(h,w2) + b2


#make the softmax function

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x),dim =1).view(-1,1)

probablities = softmax(out)

print(probablities.shape)




