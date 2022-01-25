import torch

#x = torch.empty(2,3 ,2,3)
#print(x)

#a one dimensional data type
#z= torch.tensor([2,3,4,6])
#print(z)

x = torch.rand(2,2)
y = torch.rand(2,2)


print(x)
print(y)

y.add_(x) # for addition

z = torch.add(x,y)
z= x+y

print(y)