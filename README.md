# Predicting Real Estate Prices with Machine Learning
Using different machine learning methods (SVM, decision tree, random forest, artificial neural network) to predict 
prices of real estate. You can find the [data](data/train.csv), along with its [description](data/data_description.txt) in 
the [data folder](data).

### Topics
Python, Machine learning, SVM, Decision tree, Random forest, Artificial neural network, Real estate

### Packages
```
pandas
numpy
scikit-learn
sklearn-evaluation
plotly
matplotlib
```

## Data preparation
Firstly, I loaded previously encoded (dummy) data. As part of the preparation, I cut 10% of the most correlated 
data and scaled the remaining data (except for the output) using MinMaxScaler.

## Correlation matrix
The images below show the correlation matrix before and after removing the most correlated data.

<img src="images\saved\corr_matrix.png" width="400"/> <img src="images\saved\corr_matrix_final.png" width="400"/>

## Training the models
### Grid search
I first trained all models using grid search and cross-validation, specifically using the GridSearchCV function from 
the sklearn package. Along with GridSearchCV, I also used model implementations from sklearn.
### Decision tree
I used the DecisionTreeRegressor() function to train the decision tree.
Using grid search, I tried different values for the 'criterion', 'max_depth' and 'max_features' parameters.

<img src="images\saved\dtr_params.png"/>

The best model turned out to be {'criterion': 'poisson', 'max_depth': 20, 'max_features': 150}.
It had a success rate of 0.7124 on the training set and 0.7023 on the test set. (Mean squared error = 1 769 984 646).


In the following image, you can see a visualization of a tree with a maximum depth of 5.
<img src="images\saved\tree_visual.png"/>
[tree_visual.png](images\saved\tree_visual.png)

### SVM
I used the SVR() function to train the SVM. Using grid search, I tried different values for the 'kernel', 'C' and 
'gamma' parameters.

<img src="images\saved\svm_params.png"/>

The best model turned out to be {'kernel': 'linear', 'C': 6000, 'gamma': 50}. It had a success rate of 0.8272 on the 
training set and 0.9037 on the test set. (Mean squared error = 572 230 191). This made it the most successful of our 
models.

#### Grid search results:
<img src="images\saved\svm_gs_fin.png" width="400"/> <img src="images\saved\svm_gs_lin_c-6000.png" width="400"/>

At first glance, the poly kernel seemed to be the best, but at high values of the C parameter, the linear kernel outperformed it.

### Random forest
I used the function RandomForestRegressor() to train the Random forest. Using grid search, I tried different values for 
the parameters 'n_estimators', 'criterion', 'max_depth', 'min_samples_split' and 'max_features'.

<img src="images\saved\rfr_params.png">

The best model turned out to be {'n_estimators': 200, 'criterion': 'poisson', 'max_depth': None, 'min_samples_split': 2,
'max_features': 0.5}. It had a success rate of 0.8331 on the training set and 0.8834 on the test set. (Mean squared error = 693 173 451).

#### Importance of input features
Below you can see the most important (left) and least important (right) input features.
<img src="images\saved\feature_importance_best.png" width="400"> <img src="images\saved\feature_importance_worst.png" width="400">

## Analysis of residuals
#### Decision tree:
<img src="images\saved\dtr_residuals.png" width="400">

#### SVM:
<img src="images\saved\svm_residuals.png" width="400">

#### Random forest:
<img src="images\saved\rfr_residuals.png" width="400">

## Dimension reduction
### 3 selected features
I chose 3 features, namely GarageCars, GrLivArea and YearBuilt (that is, the 3 most significant according to feature 
importance from the random forest).

<img src="images\saved\reduction_3.png">

### PCA
I minimized the set to 3 dimensions using PCA. It can be seen that the groups have a better representation and much 
fewer outliers or extreme values than with the 3 selected features.

<img src="images\saved\reduction_pca.png">

## Subset reduced to X dimensions
I first separated a subset of the 150 most significant features and then tried
reducing it to number of dimensions in the interval from 5 to 145.

<img src="images\saved\red_x.png">

After the largest dimension was the most successful, I expanded the subset to 220 elements. The best result was reached 
when reducing to 185 dimensions and thus achieved r2 success rate of 0.9045, which ultimately improved the 
decision tree model.

<img src="images\saved\red_x_best.png">

## Bonuses
### Clustering
<img src="images\saved\clustering.png">

### Cluster training
<img src="images\results\cluster_training.png">

### Artificial neural network
As a bonus, I used the MLPRegressor() function to train an Artificial neural network.

<img src="images\saved\mlp_params.png">

The best model I found was {'hidden_layer_sizes': [100], 'activation': 'identity', 'solver':
'lbfgs', 'max_iter': 3000}. It had a success rate of 0.6794 in training and 0.8942 in testing
sets. (Mean squared error = 629 003 268).

<img src="images\results\mlp.png">
