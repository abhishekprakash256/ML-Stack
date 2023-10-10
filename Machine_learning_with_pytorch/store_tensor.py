"""
The file to store the tensor
"""

import torch

# Create a sample PyTorch tensor
tensor_to_save = torch.tensor([1, 2, 3])

# Specify a file path for storing the tensor
file_path = "tensor.pt"

# Save the tensor to the specified file
torch.save(tensor_to_save, file_path)

loaded_tensor = torch.load(file_path)

# Now, 'loaded_tensor' contains the PyTorch tensor
print("Loaded Tensor:", loaded_tensor)