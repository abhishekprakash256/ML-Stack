import torch
import pandas as pd
import numpy as np

a = torch.tensor([1,2,3])

b = torch.tensor([4,5,6])

mat_1 = torch.tensor([[1,2],[3,4]])

mat_2 = torch.tensor([[5,6],[]])

c = a.mul(b)

d = a.dot(b)

print(c)

print(d)
