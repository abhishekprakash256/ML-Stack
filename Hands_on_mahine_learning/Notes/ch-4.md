## Ch - 4 
### Training Models

### Linear regression 
```
Equation 4-1. Linear Regression model prediction
y = θ0 + θ1x1 + θ2x2 + ⋯ + θnxn
• ŷ is the predicted value.
• n is the number of features.
• xi is the ith feature value.
• θj is the jth model parameter (including the bias term θ0 and the feature weights
θ1, θ2, ⋯, θn).

```

```
Equation 4-2. Linear Regression model prediction (vectorized form)
y = hθ(x) = θT.x
• θ is the model’s parameter vector, containing the bias term θ0 and the feature
weights θ1 to θn.
• θT is the transpose of θ (a row vector instead of a column vector).
• x is the instance’s feature vector, containing x0 to xn, with x0 always equal to 1.
• θT · x is the dot product of θT and x.
• hθ is the hypothesis function, using the model parameters θ.

```


### MSE

![mse equation](mse.png)



![mse equation](normal.png)



