"""
make a torch model to find the accuracy of the model and trained on a dummy data 
the model we are are making is linear model
"""

#imports 
import torch as th 
from torch import nn 


#make the parameters 
WEIGHT = 0.7
BIAS = 0.3 
EPOCHS = 500 
th.manual_seed(5)


class Data_handling():
    """
    A class for data handling operations, including data generation and train-test splitting.

    Attributes:
    X (Tensor): Input data tensor.
    Y (Tensor): Target data tensor.
    X_train (Tensor): Input data for training.
    y_train (Tensor): Target data for training.
    X_test (Tensor): Input data for testing.
    y_test (Tensor): Target data for testing.

    Methods:
    make_data(): Generates synthetic data with specified start, end, and step values.
    train_test_split(): Splits the generated data into training and testing sets.
    """
    
    def make_data(self):
        """
        Generate synthetic data.

        Generates synthetic data using specified start, end, and step values.
        The data consists of input values (X) and corresponding target values (Y),
        calculated based on WEIGHT and BIAS.

        Returns:
        None
        """
        start = 0
        end = 2 
        step = 0.02

        self.X = th.arange(start, end, step).unsqueeze(dim=1)
        self.Y = WEIGHT * self.X + BIAS
    
    def train_test_split(self):
        """
        Split the generated data into training and testing sets.

        Splits the generated data into training and testing sets based on a predefined split ratio (80% train, 20% test).

        Returns:
        None
        """
        train_split = int(0.8 * len(self.X))

        self.X_train, self.y_train = self.X[:train_split], self.Y[:train_split]
        self.X_test, self.y_test = self.X[train_split:], self.Y[train_split:]



class Linear_Model(nn.Module):
    """
    A simple linear regression model implemented as a PyTorch module.

    Attributes:
    linear_layer (nn.Linear): The linear transformation layer of the model.

    Methods:
    forward(x): Performs a forward pass through the model.
    """

    def __init__(self):
        """
        Initialize the Linear_Model.

        Creates a linear transformation layer with one input feature and one output feature.

        Returns:
        None
        """
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        """
        Forward pass through the linear regression model.

        Args:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor produced by the linear regression model.
        """
        return self.linear_layer(x)


class Test_and_train:

	def train_and_test_model(self):
		"""
		Train and test a linear regression model using synthetic data.

		This function creates an instance of the `Data_handling` class to generate synthetic data and split it into
		training and testing sets. It also creates an instance of the `Linear_Model` class for linear regression.
		The training process includes defining a loss function (L1Loss), an optimizer (SGD), and performing training epochs.
		During training, it prints the loss and test loss every 10 epochs.

		Returns:
		None
		"""
		
		# Create a data instance
		data = Data_handling()

		# Generate and split the data
		data.make_data()
		data.train_test_split()

		# Create a linear regression model instance
		LR_model = Linear_Model()

		# Define the loss function
		loss_fn = nn.L1Loss()

		# Define the optimizer
		optimizer = th.optim.SGD(params=LR_model.parameters(), lr=0.01)

		for epoch in range(EPOCHS):
			LR_model.train()
			y_pred = LR_model(data.X_train)
			loss = loss_fn(y_pred, data.y_train)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			LR_model.eval()
			with th.inference_mode():
				test_pred = LR_model(data.X_test)
				test_loss = loss_fn(test_pred, data.y_test)

			if epoch % 10 == 0:
				print(f"Epoch {epoch}: Training Loss: {loss}, Test Loss: {test_loss}")
				print(LR_model.state_dict())

		return LR_model.state_dict()




if __name__ == "__main__":
	
	trainer = Test_and_train()
	param = trainer.train_and_test_model()
	model = Linear_Model()
	model.load_state_dict(param)

	




