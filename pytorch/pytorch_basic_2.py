import torch

x=torch.tensor([1,2,3])

z= torch.arange(0,18).reshape(3,6)

y= torch.rand(4,3)

s= torch.randn(4,3)

g= torch.randint(0,10,(5,5))

torch.manual_seed(42)

f = torch.rand(2,3)

print(x)

print(z)

print(y)

print(s)

print(g)

print(f)

print(x.shape)