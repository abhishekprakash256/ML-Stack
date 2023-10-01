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

	print(list(LR_model.parameters()))

