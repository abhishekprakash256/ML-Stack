
"""
store the circle data from the scikit learn in torch format 
"""

#imports 
import torch as th


# Specify a file path for storing the tensor
x_file_path = "x.pt"
y_file_path = "y.pt"


x = th.load(x_file_path)
y = th.load(y_file_path)

