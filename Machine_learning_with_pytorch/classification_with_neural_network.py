"""
The classification problem using a neural network 
"""

#imports 
import torch as th
from sklearn.datasets import make_circles 
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#constants
EPOCHS = 500 
th.manual_seed(42)







class Data:
	def make_data(self):
		"""
		The method to make the data set and load the values
		"""

		data = make_circles(n_samples=1000, shuffle=True, noise=None, random_state=None, factor=0.8)
		self.X,self.y = data
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

		self.X_train = th.tensor(self.X_train)
		self.X_test = th.tensor(self.X_test)
		self.y_train = th.tensor(self.y_train)
		self.y_test = th.tensor(self.y_test)


class Visualization:
	def visualize_data(self):
		"""
		The method to visualize the data in pyplot
		"""

		







		





if __name__ == "__main__":
	data = Data()
	data.make_data()