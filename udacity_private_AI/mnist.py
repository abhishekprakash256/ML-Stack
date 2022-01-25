#imports
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)




#making the model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #the layers 
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.out = nn.Linear(64,10)

    def forward(self,x):
        #flatten the tensor 
        x = x.view(x.shape[0],-1)

        x = F.relu(self.fc1(x))  #the hidden layer
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = F.softmax(self.out(x), dim = 1) #the output 

        return x



model = Network()

#optimizer and loss

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


#training the model 

epochs = 5

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
