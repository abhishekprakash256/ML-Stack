import numpy as np
import pandas as pd
from numpy.random import randn

mat = randn(5,4)

print(mat)

new = pd.DataFrame(data =mat)

new_2 = pd.DataFrame(data = mat , index= "A B C D E".split(), columns = "W X Y Z".split())

print(new)

new_2['NEW'] = new_2[['W','X']]

print(new_2)
