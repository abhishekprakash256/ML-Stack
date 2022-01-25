import torch
import pandas as pd
import numpy as np

#make a tesnor here
x = torch.tensor(2.0, requires_grad = True)

#create the function here
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1


y.backward() #this do the first derivative
print(y)
print(x)

print(x.grad) #plugging in the value of x

