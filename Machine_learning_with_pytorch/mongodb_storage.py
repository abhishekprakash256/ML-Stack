"""
Store created array data in mongodb 
"""

#imports 

import torch
from pymongo import MongoClient


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
client.close()

