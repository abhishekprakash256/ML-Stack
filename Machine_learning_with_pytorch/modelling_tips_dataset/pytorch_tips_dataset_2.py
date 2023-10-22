"""
linear regression with tips dataset , preparing the dataset as one hot encoding as well as label encoding 
"""

#imports 
import pandas as pd
import torch as th
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


#data file path
FILE_PATH = "../datasets/tips_full.csv"
EPOCHS = 1000



class Data_Prepration_OneHot:
    """
    class to prepare the data as 
    
    """

    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


    def make_data(self):
        """
        get the data and split in the data in X and y 
        """
        self.df = pd.read_csv(FILE_PATH)
        self.y = self.df["tip"].astype(float)


        #make the train set 
        col = ["total_bill","tip","sex","smoker","day","time","size"]
        self.X = self.df[col]
        self.X = pd.get_dummies(self.X).astype(float)


    def split_data(self):
        """
        split the data in test and train
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.33, random_state=42)

        self.X_train = th.tensor(self.X_train.values,dtype= th.float32)
        self.X_test = th.tensor(self.X_test.values, dtype= th.float32)
        self.y_train = th.tensor(self.y_train.values,  dtype= th.float32)
        self.y_test = th.tensor(self.y_test.values,  dtype= th.float32)



class Data_Prepration_Labellig:
    """
    class to prepare the data as 
    
    """

    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


    def make_data(self):
        """
        get the data and split in the data in X and y 
        """
        self.df = pd.read_csv(FILE_PATH)
        self.y = self.df["tip"].astype(float)


        #make the train set 
        col = ["total_bill","tip","sex","smoker","day","time","size"]
        self.X = self.df[col]

        #do the label encoding 
        lab = LabelEncoder()

        label_col = ["sex","smoker","day","time"]

        for item in label_col:

            self.X[item] = lab.fit_transform(self.X[item])


    def split_data(self):
        """
        split the data in test and train
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.33, random_state=42)

        self.X_train = th.tensor(self.X_train.values,dtype= th.float32)
        self.X_test = th.tensor(self.X_test.values, dtype= th.float32)
        self.y_train = th.tensor(self.y_train.values,  dtype= th.float32)
        self.y_test = th.tensor(self.y_test.values,  dtype= th.float32)



#make the pytorch regression model

class Liner_Model(nn.Module):
	"""
	Make a linear model
	"""

	def __init__(self, in_features):
		super(Liner_Model, self).__init__()
		self.linear1 = nn.Linear(in_features, 1)  #input dims =  5 , output dims= 1

	def forward(self, x):
		x = self.linear1(x)

		return x


def pytorch_train_test_onehot():
    """
    train and test for pytorch one hot encoding 
    """

    model = Liner_Model(in_features=7)

    data = Data_Prepration_Labellig()
    data.make_data()
    data.split_data()

    #loss function 
    loss_fn = nn.L1Loss()

    # Define the optimizer	
    optimizer = th.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.1)

    #the loop for trainer
    for epoch in range(EPOCHS):
        
        optimizer.zero_grad()
        train_predictions = model(data.X_train).flatten()
        loss = loss_fn(train_predictions, data.y_train)

        loss.backward()
        optimizer.step()

        test_predictions = model(data.X_test).flatten()
        test_loss = loss_fn(test_predictions, data.y_test)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    test_pred = model(data.X_train[0])

    print(data.X_train[0])
    print(test_pred)




def pytorch_train_test_labelling():
    """
    train and test for pytorch one hot encoding 
    """

    model = Liner_Model(in_features=13)

    data = Data_Prepration_OneHot()
    data.make_data()
    data.split_data()

    #loss function 
    loss_fn = nn.L1Loss()

    # Define the optimizer	
    optimizer = th.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.1)

    #the loop for trainer
    for epoch in range(EPOCHS):
        
        optimizer.zero_grad()
        train_predictions = model(data.X_train).flatten()
        loss = loss_fn(train_predictions, data.y_train)

        loss.backward()
        optimizer.step()

        test_predictions = model(data.X_test).flatten()
        test_loss = loss_fn(test_predictions, data.y_test)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    test_pred = model(data.X_train[0])

    print(data.X_train[0])
    print(test_pred)


def sklearn_train_test_onehot():
	"""
	The fumction for test and train    
	""" 

	#data handling 
	model = LinearRegression()
	data_onehot = Data_Prepration_OneHot()
	data_onehot.make_data()
	data_onehot.split_data()

	model.fit(data_onehot.X_train,data_onehot.y_train)
	y_pred = model.predict(data_onehot.X_test)

	mse = mean_squared_error(data_onehot.y_test, y_pred, squared= False)
	print(mse)


def sklearn_train_test_labelling():
	"""
	The fumction for test and train    
	""" 

	#data handling 
	model = LinearRegression()
	data_labelling = Data_Prepration_Labellig()
	data_labelling.make_data()
	data_labelling.split_data()

	model.fit(data_labelling.X_train,data_labelling.y_train)
	y_pred = model.predict(data_labelling.X_test)

	mse = mean_squared_error(data_labelling.y_test, y_pred, squared= False)
	print(mse)




if __name__ == "__main__":

    sklearn_train_test_onehot()
    sklearn_train_test_labelling()

    print("one hot encoding training")
    pytorch_train_test_onehot()

    print("labelling encoding training")
    pytorch_train_test_labelling()
