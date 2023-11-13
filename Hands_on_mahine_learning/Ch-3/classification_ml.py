"""
to make the problem a classification problem 
"""

from sklearn.datasets import load_digits
import seaborn as sns


digits = load_digits()

X, y = digits.data, digits.target

print(X.shape)

