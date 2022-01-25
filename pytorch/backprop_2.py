import torch
import pandas as pd
import numpy as np

x= torch.tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad = True)

y = 3*x + 2

z = 2*y**2

a = z.mean()

print(y)

print(z)

a.backward()

print(x.grad)

print(a)
