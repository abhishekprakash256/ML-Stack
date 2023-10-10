"""
Store created array data in mongodb 
"""

#imports 

import torch
from pymongo import MongoClient
import numpy as np

# Create a MongoClient and connect to your MongoDB server
client = MongoClient("mongodb://localhost:27017")

# Choose a database and collection
db = client["mydatabase"]
collection = db["mycollection"]

# Create a Torch tensor
tensor = torch.tensor([1, 2, 3])

# Convert the Torch tensor to a NumPy array
numpy_array = tensor.numpy()

# Convert the NumPy array to bytes for storage
tensor_bytes = numpy_array.tobytes()

# Create a document to store the tensor bytes
document = {"tensor_data": tensor_bytes}

# Insert the document into the collection
collection.insert_one(document)

# Close the MongoDB connection when done

document = collection.find_one({})

if document:
    # Retrieve the tensor data from the document
    tensor_bytes = document.get("tensor_data", None)

    if tensor_bytes:
        # Convert the bytes back to a NumPy array
        numpy_array = np.frombuffer(tensor_bytes, dtype=np.float32)  # Adjust dtype if needed

        # Convert the NumPy array back to a Torch tensor
        tensor = torch.from_numpy(numpy_array)

        # Now you have your Torch tensor
        print("Retrieved Torch Tensor:", tensor)



client.close()

#--retrive this data from mongodb 
