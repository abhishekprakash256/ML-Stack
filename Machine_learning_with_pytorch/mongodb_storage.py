"""
Store created array data in mongodb 
"""

#imports 

import torch as th 
from pymongo import MongoClient


import pymongo

# Replace these with your MongoDB connection details
mongo_uri = "mongodb://localhost:27017"  # Connection URI
database_name = ""      # Database name
collection_name = ""        # Name of the new collection

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)

# Access the database
db = client[sample_data]

# Create a new collection
db.create_collection(test_data)

# Close the MongoDB connection when done
client.close()

print(f"Collection '{collection_name}' created successfully.")
