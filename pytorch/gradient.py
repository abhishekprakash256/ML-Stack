#gradient calculation with autograd 
import torch 

#make a tensor 
x = torch.tensor([1.0,2.0,3.0], requires_grad = True)

y = x+2 

z = y*y*2


print(x)

print(y)

print(z)

z = z.sum()

print(z)

z.backward()

#print(z)

print(x.grad)
