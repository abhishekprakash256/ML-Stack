"""
usingg the xg boost for the regression
"""

import pandas as pd 
import torch as th
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor  
from helper.helper_function import Data_Prepration_OneHot,Data_Prepration_Labellig


#file locations 
FILE_PATH = "./Machine_learning_with_pytorch/datasets/tips_full.csv"

EPOCH = 1000

#make the different models 

#linear regression
linear_regression = LinearRegression()

#Random forest model 
random_forest = RandomForestRegressor()

#xgboost 
XG_boost = XGBRFRegressor(n_estimators=100, subsample=0.9, colsample_bynode=0.2)

#make the Decission tree
Decision_tree_regressor = DecisionTreeRegressor()


data_onehot = Data_Prepration_OneHot()
data_onehot.make_data(FILE_PATH)
data_onehot.split_data()

data_labelling = Data_Prepration_Labellig()
data_labelling.make_data(FILE_PATH)
data_labelling.split_data()





def train_test_one_hot():
    """
    train and test the one hot encoding model
    """

    #fitting the model 

    linear_regression.fit(data_onehot.X_train,data_onehot.y_train)
    random_forest.fit(data_onehot.X_train,data_onehot.y_train)
    XG_boost.fit(data_onehot.X_train,data_onehot.y_train)
    Decision_tree_regressor.fit(data_onehot.X_train,data_onehot.y_train)

 
    #calulate loss 
    y_pred_regression = linear_regression.predict(data_onehot.X_test)
    mse_regression = mean_squared_error(data_onehot.y_test, y_pred_regression, squared= False)
    print(mse_regression)

    #calulate loss 
    y_pred_forest = random_forest.predict(data_onehot.X_test)
    mse_forest = mean_squared_error(data_onehot.y_test, y_pred_forest, squared= False)
    print(mse_forest)


    #calulate loss 
    y_pred_xgboost = XG_boost.predict(data_onehot.X_test)
    mse_xgboost = mean_squared_error(data_onehot.y_test, y_pred_xgboost, squared= False)
    print(mse_xgboost)

    #calulate loss 
    y_pred_decision_tree = Decision_tree_regressor.predict(data_onehot.X_test)
    mse_dst = mean_squared_error(data_onehot.y_test, y_pred_decision_tree, squared= False)
    print(mse_dst)
 



def train_test_labelling():
    """
    train and test the one hot encoding model
    """

    #fitting the model 

    linear_regression.fit(data_labelling.X_train,data_labelling.y_train)
    random_forest.fit(data_labelling.X_train,data_labelling.y_train)
    XG_boost.fit(data_labelling.X_train,data_labelling.y_train)
    Decision_tree_regressor.fit(data_labelling.X_train,data_labelling.y_train)

 
    #calulate loss 
    y_pred_regression = linear_regression.predict(data_labelling.X_test)
    mse_regression = mean_squared_error(data_labelling.y_test, y_pred_regression, squared= False)
    print(mse_regression)

    #calulate loss 
    y_pred_forest = random_forest.predict(data_labelling.X_test)
    mse_forest = mean_squared_error(data_labelling.y_test, y_pred_forest, squared= False)
    print(mse_forest)


    #calulate loss 
    y_pred_xgboost = XG_boost.predict(data_labelling.X_test)
    mse_xgboost = mean_squared_error(data_labelling.y_test, y_pred_xgboost, squared= False)
    print(mse_xgboost)

    #calulate loss 
    y_pred_decision_tree = Decision_tree_regressor.predict(data_labelling.X_test)
    mse_dst = mean_squared_error(data_labelling.y_test, y_pred_decision_tree, squared= False)
    print(mse_dst)






if __name__ == "__main__":
    train_test_one_hot()
    train_test_labelling()