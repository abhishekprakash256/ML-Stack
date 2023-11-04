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


- Try the different metrics of the machine learning data to feed before into the model as playaround with different combo


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



