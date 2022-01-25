import torch
import numpy as np

df = np.arange(1,10)

#convert to a tensor

df_torch = torch.from_numpy(df)

df_torch_new = torch.as_tensor(df)

print(df)

print(df_torch)

print(df_torch_new)