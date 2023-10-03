"""
The linear regresssion model in pytorch
"""

#imports 

import torch as th 
from torch import nn 



class Data():
	def __init__(self):

		self.weight = 0.7
		self.bias = 0.3



	def make_data(self):
		"""
		to generate the for trainning
		"""

		start = 0
		end = 1 
		step = 0.02

		self.X = th.arange(start,end,step).unsqueeze(dim=1)

		self.Y = self.weight*self.X + self.bias

		#make the test and train data set

		train_split = int(0.8*len(self.X))

		self.X_train , self.y_train = self.X[:train_split],self.Y[:train_split]
		self.X_test , self.y_test = self.X[train_split:],self.Y[train_split:]



class Linear_regression_model(nn.Module):
	"""
	Make the model from the linear regressor of the module

	"""
	def __init__(self):
		super().__init__()
		self.weights  = nn.Parameter(th.randn(1,requires_grad=True, dtype= th.float))

		self.bias = nn.Parameter(th.randn(1,requires_grad=True, dtype= th.float))


	def forward(self,x : th.Tensor):

		return self.weights * x + self.bias 


if __name__=="__main__":
	data = Data()

	data.make_data()

	LR_model = Linear_regression_model()

	param =list(LR_model.parameters())

	print(LR_model.state_dict())


	loss_fn = nn.L1Loss()

	optimizer = th.optim.SGD(params = LR_model.parameters(), lr = 0.01) 


	#make the training loop 

	epochs = 10 

	for epoch in range(epochs):

		print(epoch)
		
		LR_model.train()

		#forward pass 
		y_pred = LR_model(data.X_train)

		#calculate loss 

		loss = loss_fn(y_pred, data.y_train)

		#optimize 
		optimizer.zero_grad()

		#perform backprop 
		loss.backward()

		#perfrom gradient descent 
		optimizer.step()

		#model evaluation

		LR_model.eval()

		print(LR_model.state_dict())