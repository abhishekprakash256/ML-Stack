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

		sample_data = make_circles(n_samples=1000, shuffle=True, noise=None, random_state=None, factor=0.8)
		self.X,self.y = sample_data
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

		self.X_train = th.tensor(self.X_train).float()
		self.X_test = th.tensor(self.X_test).float()
		self.y_train = th.tensor(self.y_train).float()
		self.y_test = th.tensor(self.y_test).float()



class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		# Define layers with float data type
		self.linear1 = nn.Linear(2, 16)
		#self.linear2 = nn.Linear(16, 32)
		#self.linear3 = nn.Linear(32, 16)
		self.linear4 = nn.Linear(16, 1)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.linear1(x)
		#x = self.linear2(x)
		#x = self.linear3(x)
		x = self.linear4(x)
		x = self.sigmoid(x)
		x = th.floor(x)
 
		return x



def train_and_test():

	"""
	The funcition to train and test the model 
	"""
	data = Data()

	#make they data
	data.make_data()

	model = Model()

	# Define the loss function
	#loss_fn = nn.L1Loss()

	loss_fn = nn.BCEWithLogitsLoss()

	# Define the optimizer
	optimizer = th.optim.SGD(params=model.parameters(), lr=0.001)


	#------------------debug the code ---------------------------#

	"""
	print(data.X_train[0:5])
	print(data.y_test[0:5])

	print(model.parameters)

	
	test_tensor = th.tensor([0.4,0.555])	

	print(test_tensor)

	y_pred = model(test_tensor)

	print(y_pred)
	
	"""
	
	
	#the loop for trainer
	for epoch in range(EPOCHS):
		model.train()
		y_pred = model(data.X_train)

		#print(y_pred)

		#print("-------------y_pred______done")

		#print(y_pred.shape)
		#print(data.y_train.shape)


		loss = loss_fn(y_pred, data.y_train.reshape(800,1))

		#print(loss)

		#print("_________loss_______done")

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		model.eval()
		with th.inference_mode():
			test_pred = model(data.X_test)
			test_loss = loss_fn(test_pred, data.y_test.reshape(200,1))

		if epoch % 10 == 0:
			print(f"Epoch {epoch}: Training Loss: {loss}, Test Loss: {test_loss}")


	return model.state_dict()
	
	
if __name__ == "__main__":

	train_and_test()