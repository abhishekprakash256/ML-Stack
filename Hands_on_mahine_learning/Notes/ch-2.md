## Ch-2 

### The performance measures
- The RMSE is used
- The MSE is used
- The MAE is used

### The generalization error 
 When you estimate the generalization error using the test
set, your estimate will be too optimistic and you will launch a system that will not
perform as well as expected. This is called data snooping bias.


- The sampling bias can produse bisaing in dataset
- For example, the US population is com‐
posed of 51.3% female and 48.7% male, so a well-conducted survey in the US would
try to maintain this ratio in the sample: 513 female and 487 male. This is called strati‐
fied sampling: the population is divided into homogeneous subgroups called strata,
and the right number of instances is sampled from each stratum to guarantee that the
test set is representative of the overall population


- Try the different metrics of the machine learning data to feed before into the model as playground with different combo


### Data processsing 
- Cleaning of the data
- Imputation in the data
- One hot encoder and LabelBinarizer
- Feature scaling 
    - min-max
    scaling and standardization.
    - Normilization fix into the range of 0 to 1 (min max) 
    -  standartazition - Standardization is quite different: first it subtracts the mean value (so standardized
    values always have a zero mean), and then it divides by the variance so that the result‐
    ing distribution has unit variance. Unlike min-max scaling, standardization does not
    bound values to a specific range, which may be a problem for some algorithms (e.g.,
    neural networks often expect an input value ranging from 0 to 1).
    - Sklearn provides pipeline featues

### The model tuining 
The sklearn porvides grid searching to fine tune the model faster 

### Scattering tool to find the correlations in matrix
- We can use a correlation matrix to summarize a large data set and to identify patterns and make a decision according to it. We can also see which variable is more correlated to which variable, and we can visualize our results.
- pandas scattering tool can be used to find the realtions in the features 


### Data prepration 
- This will allow you to reproduce these transformations easily on any dataset (e.g.,
the next time you get a fresh dataset).
- You will gradually build a library of transformation functions that you can reuse
in future projects.
- You can use these functions in your live system to transform the new data before
feeding it to your algorithms.


#### Data cleaning steps 
- Get rid of the corresponding districts.
- Get rid of the whole attribute.
- Set the values to some value (zero, the mean, the median, etc.)
- You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna()
methods:

#### Imputation 
- fill the value with median , mode , fixed.

#### Encoding
- Labelencoder 
- onehot encoding
```
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
```
### Custom transformers can be written on sklearn as well 
- using transformers 
### Feature scaling 
- The value are scaled to fit in certain value, standardization and min max scaling. The feature scaling can be done to capture the outcast in the data. 



