"""
Store created array data in mongodb 
"""

#imports 

import torch as th 
from pymongo import MongoClient


import pymongo

# Replace these with your MongoDB connection details
mongo_uri = "mongodb://localhost:27017"  # Connection URI
database_name = "Test_Data_Base"      # Database name
collection_name = "Test_Collection"        # Name of the new collection

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)

# Access the database
db = client[database_name]

# Create a new collection
db.create_collection(collection_name)

# Close the MongoDB connection when done
client.close()

print(f"Collection '{collection_name}' created successfully.")
