"""
linear regression with tips dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



#data file path
FILE_PATH = "./datasets/tips.csv"
EPOCHS = 50

class Data:

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
		self.df = pd.get_dummies(self.df, columns=['day'])
		self.df = self.df.astype(float)
		self.df = self.df.rename(columns = {'day_Friday': 'Friday', 'day_Tuesday': 'Tuesday'})
		
		col = ["week","n_peop","bill","Friday","Tuesday"]

		self.X = self.df[col]
		self.y = self.df["tip"]

	def split_data(self):
		"""
		split the data in test and train
		"""
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.33, random_state=42)
		
		
		#self.X_train = th.tensor(self.X_train.values,dtype= th.float32)
		#self.X_test = th.tensor(self.X_test.values, dtype= th.float32)
		#self.y_train = th.tensor(self.y_train.values,  dtype= th.float32)
		#self.y_test = th.tensor(self.y_test.values,  dtype= th.float32)



def train_test():
    """
    The fumction for test and train    
    """ 

    #data handling 
    model = LinearRegression()
    data = Data()
    data.make_data()
    data.split_data()

    model.fit(data.X_train,data.y_train)
    y_pred = model.predict(data.X_test)

    mse = mean_squared_error(data.y_test, y_pred, squared= False)
    print(mse)

    print(y_pred.shape)

    print(data.y_test.shape)

    #print(model.score(y_pred,data.y_test))









if __name__ == "__main__":
	train_test()