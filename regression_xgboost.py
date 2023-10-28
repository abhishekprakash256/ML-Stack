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
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
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

#make the svm regressor
sv_regressor = SVR()

#make the gaussian regressor
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)


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
    sv_regressor.fit(data_onehot.X_train,data_onehot.y_train)
    gpr.fit(data_onehot.X_train,data_onehot.y_train)
 
    #calulate loss 
    y_pred_regression = linear_regression.predict(data_onehot.X_test)
    mse_regression = mean_squared_error(data_onehot.y_test, y_pred_regression, squared= False)
    print("linear regressor mse: ", mse_regression)
    print("linear regressor score: ",linear_regression.score(data_onehot.X_test,data_onehot.y_test))

    #calulate loss 
    y_pred_forest = random_forest.predict(data_onehot.X_test)
    mse_forest = mean_squared_error(data_onehot.y_test, y_pred_forest, squared= False)
    print("linear random forest mse: ", mse_forest)
    print("linear random forest score: ",random_forest.score(data_onehot.X_test,data_onehot.y_test))


    #calulate loss 
    y_pred_xgboost = XG_boost.predict(data_onehot.X_test)
    mse_xgboost = mean_squared_error(data_onehot.y_test, y_pred_xgboost, squared= False)
    print("linear xg boost mse: ", mse_xgboost)
    print("linear xg boost score: ",XG_boost.score(data_onehot.X_test,data_onehot.y_test))

    #calulate loss 
    y_pred_decision_tree = Decision_tree_regressor.predict(data_onehot.X_test)
    mse_dst = mean_squared_error(data_onehot.y_test, y_pred_decision_tree, squared= False)
    print("linear decision tree mse: ", mse_dst)
    print("linear decision tree score: ",Decision_tree_regressor.score(data_onehot.X_test,data_onehot.y_test))
 

    #calulate loss 
    y_pred_svr = sv_regressor.predict(data_onehot.X_test)
    mse_svr = mean_squared_error(data_onehot.y_test, y_pred_svr, squared= False)
    print("linear svr mse: ", mse_svr)
    print("linear svr score: ",sv_regressor.score(data_onehot.X_test,data_onehot.y_test))

    #calulate loss 
    y_pred_gpr = gpr.predict(data_onehot.X_test)
    mse_gpr = mean_squared_error(data_onehot.y_test, y_pred_gpr, squared= False)
    print("linear gpr  mse: ", mse_gpr)
    print("linear gpr score: ",gpr.score(data_onehot.X_test,data_onehot.y_test))




def train_test_labelling():
    """
    train and test the one labele encoding
    """

    #fitting the model 

    linear_regression.fit(data_labelling.X_train,data_labelling.y_train)
    random_forest.fit(data_labelling.X_train,data_labelling.y_train)
    XG_boost.fit(data_labelling.X_train,data_labelling.y_train)
    Decision_tree_regressor.fit(data_labelling.X_train,data_labelling.y_train)
    sv_regressor.fit(data_labelling.X_train,data_labelling.y_train)
    gpr.fit(data_labelling.X_train,data_labelling.y_train)

    #calulate loss 
    y_pred_regression = linear_regression.predict(data_labelling.X_test)
    mse_regression = mean_squared_error(data_labelling.y_test, y_pred_regression, squared= False)
    print("linear regressor mse: ", mse_regression)
    print("linear regressor score: ",linear_regression.score(data_labelling.X_test,data_labelling.y_test))

    #calulate loss 
    y_pred_forest = random_forest.predict(data_labelling.X_test)
    mse_forest = mean_squared_error(data_labelling.y_test, y_pred_forest, squared= False)
    print("linear forest mse: ", mse_forest)
    print("linear forest score: ",random_forest.score(data_labelling.X_test,data_labelling.y_test))

    #calulate loss 
    y_pred_xgboost = XG_boost.predict(data_labelling.X_test)
    mse_xgboost = mean_squared_error(data_labelling.y_test, y_pred_xgboost, squared= False)
    print("linear xgboost mse: ", mse_xgboost)
    print("linear xgboost score: ",XG_boost.score(data_labelling.X_test,data_labelling.y_test))

    #calulate loss 
    y_pred_decision_tree = Decision_tree_regressor.predict(data_labelling.X_test)
    mse_dst = mean_squared_error(data_labelling.y_test, y_pred_decision_tree, squared= False)
    print("linear dst mse: ", mse_dst)
    print("linear dst score: ",Decision_tree_regressor.score(data_labelling.X_test,data_labelling.y_test))

    #calulate loss 
    y_pred_svr = sv_regressor.predict(data_labelling.X_test)
    mse_svr = mean_squared_error(data_labelling.y_test, y_pred_svr, squared= False)
    print("linear svr mse: ", mse_svr)
    print("linear svr score: ",sv_regressor.score(data_labelling.X_test,data_labelling.y_test))

    #calulate loss 
    y_pred_gpr = gpr.predict(data_labelling.X_test)
    mse_gpr = mean_squared_error(data_labelling.y_test, y_pred_gpr, squared= False)
    print("linear gpr mse: ", mse_gpr)
    print("linear gpr score: ",gpr.score(data_labelling.X_test,data_labelling.y_test))






if __name__ == "__main__":
    print("-------- one hot --------")
    train_test_one_hot()
    print("-------- labelling dataset --------")
    train_test_labelling()