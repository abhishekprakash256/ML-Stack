import torch 


#sigmoid activation function 
def activation(x):

    return 1/(1+torch.exp(-x))


#generate the fake data 
torch.manual_seed(7)

features = torch.randn((1,5))
weights = torch.randn_like(features)

bias = torch.randn((1,1))

#passing the values 

print(activation((features*weights).sum()+bias))




