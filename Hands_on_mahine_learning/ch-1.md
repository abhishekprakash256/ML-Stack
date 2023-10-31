## Ch -1 

### What is machine learning ? 

![image-20231031003706422](/home/abhi/.config/Typora/typora-user-images/image-20231031003706422.png)

### The machine learning diagram

A computer program is said to learn from experience E with respect to some task T
and some performance measure P, if its performance on T, as measured by P, improves
with experience E.

### Use of the machine learning 

 First you would look at what spam typically looks like. You might notice that
some words or phrases (such as “4U,” “credit card,” “free,” and “amazing”) tend to
come up a lot in the subject. Perhaps you would also notice a few other patterns
in the sender’s name, the email’s body, and so on.

#### Data mining 

Applying ML techniques to dig into large amounts of data can help discover patterns
that were not immediately apparent. This is called data mining.

### Machine learning is good for

- Problems for which existing solutions require a lot of hand-tuning or long lists of
  rules: one Machine Learning algorithm can often simplify code and perform better.
- Complex problems for which there is no good solution at all using a traditional
  approach: the best Machine Learning techniques can find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
-  Getting insights about complex problems and large amounts of data

### Types of Machine learning

- Whether or not they are trained with human supervision (supervised, unsuper‐
  vised, semisupervised, and Reinforcement Learning)
- Whether or not they can learn incrementally on the fly (online versus batch
  learning)
-  Whether they work by simply comparing new data points to known data points,
  or instead detect patterns in the training data and build a predictive model, much
  like scientists do (instance-based versus model-based learning)

### The 4 major type of learning 

There are four major categories: supervised learning, unsupervised learning, semisupervised learning, and Reinforcement Learning.

### Common supervised learning Algo

- k-Nearest Neighbors
-  Linear Regression
-  Logistic Regression
-  Support Vector Machines (SVMs)
-  Decision Trees and Random Forests
-  Neural networks2

### Common Unsupervised Algo 

• Clustering
— k-Means
— Hierarchical Cluster Analysis (HCA)
— Expectation Maximization
• Visualization and dimensionality reduction
— Principal Component Analysis (PCA)
— Kernel PCA
— Locally-Linear Embedding (LLE)
— t-distributed Stochastic Neighbor Embedding (t-SNE)
• Association rule learning
— Apriori
— Eclat

### Dimension reuduction

A related task is dimensionality reduction, in which the goal is to simplify the data
without losing too much information. One way to do this is to merge several correla‐
ted features into one. For example, a car’s mileage may be very correlated with its age,
so the dimensionality reduction algorithm will merge them into one feature that rep‐
resents the car’s wear and tear. This is called feature extraction.

### Semi supervised learning 

Some algorithms can deal with partially labeled training data, usually a lot of unla‐
beled data and a little bit of labeled data. This is called semisupervised learning

### Reinforcememt learning 

Reinforcement Learning is a very different beast. The learning system, called an agent
in this context, can observe the environment, select and perform actions, and get
rewards in return (or penalties in the form of negative rewards, as in Figure 1-12). It
must then learn by itself what is the best strategy, called a policy, to get the most
reward over time. A policy defines what action the agent should choose when it is in a
given situation.

### Challanges in Machine learning

#### Bad data Problems

- Insufficient Quantity of Training Data
- Nonrepresentative Training Data
- Irrelevant Features
  - A critical part of the success of a Machine Learning project is coming up with a
    good set of features to train on. This process, called feature engineering, involves:
  - Feature selection: selecting the most useful features to train on among existing
    features.
  - Feature extraction: combining existing features to produce a more useful one (as
    we saw earlier, dimensionality reduction algorithms can help).
  - Creating new features by gathering new data.

#### Bad Algo Problems

- Overfitting the Training Data
  - Constraining a model to make it simpler and reduce the risk of overfitting is called
    regularization.
- Underfitting the Training Data
  -  Selecting a more powerful model, with more parameters
  -  Feeding better features to the learning algorithm (feature engineering)
  - Reducing the constraints on the model (e.g., reducing the regularization hyper‐
    parameter)

To avoid “wasting” too much training data in validation sets, a common technique is
to use cross-validation: the training set is split into complementary subsets, and each
model is trained against a different combination of these subsets and validated
against the remaining parts. Once the model type and hyperparameters have been
selected, a final model is trained using these hyperparameters on the full training set,
and the generalized error is measured on the test set.


















