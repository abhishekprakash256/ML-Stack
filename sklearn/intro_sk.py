"""
make the sklearn file for the pipeline 
"""

from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
import pandas as pd


"""


mod = KNeighborsRegressor()

mod.fit(X,y)

pred = mod.predict(X)

print(pred)

"""

X,y = fetch_california_housing(return_X_y=True)

pipe = Pipeline([("scale",StandardScaler()),("model", KNeighborsRegressor(n_neighbors =1 ))])


mod = GridSearchCV(estimator = pipe,param_grid = {'model__n_neighbors':[1,2,3,4,5,6]},cv =3)

mod.fit(X,y)

print(pd.DataFrame(mod.cv_results_))