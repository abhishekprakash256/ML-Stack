"""
The experiment to load the data from an array 
"""

import torch as th 

x = th.rand(50)
print(x)

for i in range(10):

    print(x[i*5:(i+1)*5])