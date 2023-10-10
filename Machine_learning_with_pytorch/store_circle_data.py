"""
store the circle data from the scikit learn in torch format 
"""

#imports 
import torch as th
from sklearn.datasets import make_circles 
from sklearn.model_selection import train_test_split

#sample_data = make_circles(n_samples=30, shuffle=True, noise=None, random_state=None, factor=0.8)

sample_data = make_circles(n_samples=3000000, shuffle=True, noise=None, random_state=None, factor=0.8)

x,y = sample_data

x = th.tensor(x,dtype=th.float64)
y = th.tensor(x,dtype=th.float64)

# Specify a file path for storing the tensor
x_file_path = "x.pt"
y_file_path = "y.pt"

# Save the tensor to the specified file
torch.save(tensor_to_save, file_path)