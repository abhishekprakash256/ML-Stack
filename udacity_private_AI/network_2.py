#make a network of two units 
import torch 

features = torch.randn((1,3)) #inputs

#make the two h1 and h2

w1 = torch.randn((1,1)) #weights 1 
w2 = torch.randn((1,1)) #weights 2

b1 = torch.randn((1,1)) #bias 1
b2 = torch.randn((1,1)) #bias 2


#sigmoid activation function 
def activation(x):

    return 1/(1+torch.exp(-x))

# for h1
h1 = torch.mul(features,w1) + b1
h2 = torch.mul(features,w2) + b1

h1_out = activation(h1)
h2_out = activation(h2)
