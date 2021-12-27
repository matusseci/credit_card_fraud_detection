# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:46:01 2021

Going through the code in this Kaggle kernel
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

Skills used: 
    - scikit learn
    - imbalanced datasets
    - some neural nets? 
    - data cleaning and basic viz

@author: Matúš
"""

### Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # plotting library
import matplotlib.patches as mpatches # plotting dim reductiosn results
import time

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

### Read in the data

'''
The features in the data are standardized and without names
for privacy reasons.
'''

df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.describe())  # like R summary()

# Check for null values
print(df.isnull().sum())

# Check the column names
print(df.columns)

# Check the distribution of classes (fraud vs non-fraud)
print('No frauds:',  round(df['Class'].value_counts()[0]/len(df) * 100, 2), '% of the dataset')
print('Frauds:',  round(df['Class'].value_counts()[1]/len(df) * 100, 2), '% of the dataset')

colors = ['#0101DF', '#DF0101']
sns.countplot('Class', data = df, palette = colors)
plt.title('Class Distribution \n (0: No Fraud || 1: Fraud)', fontsize = 14)

# The distribution is very skewed towards non-fraudulent transcations. 
# we will have to consider this when building classifiers. 

# Plot histograms for:
# 1. Transaction amount
# 2. Transaction time

fig, ax = plt.subplots(1, 2, figsize = (18, 4))  # divides the plot panel into two subparts

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.histplot(amount_val, ax = ax[0], color = 'r')
ax[0].set_title('Distribution of Transaction Amount', fontsize = 14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.histplot(time_val, ax = ax[1], color = 'b')
ax[1].set_title('Distribution of Transaction Time', fontsize = 14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


### Scaling the features and sub-sampling

# Using Standard scaler (aka standardization) - this is not correct though? 
# or using Robust scaler

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

# The reshape(-1, 1) changes the shape into 1 column and an appropriate number of rows (indicated by -1)
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis = 1, inplace = True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis = 1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!
print(df.head())

### Splitting the dataset 

X = df.drop('Class', axis = 1)  # assign X to be all the features without target class
y = df['Class']  # y is only the target class 

sss = StratifiedKFold(n_splits = 5, random_state = None, shuffle = False)  # defien the function for stratified k-fold validation

for train_index, test_index in sss.split(X, y):  # run the function from above
    print('Train:', train_index, 'Test:', test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    
# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

### Random under-sampling

df = df.sample(frac = 1) # ???? what is this doing? 

# Extract the same number of fraud and non-fraud classes (492)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle df rows
new_df = normal_distributed_df.sample(frac = 1, random_state = 42) # random_state is the same as random seed
new_df.head()

# Look at the data division
print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot(x = 'Class', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

### Correlation matrix (with subsampled df)

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap = 'coolwarm_r', annot_kws = {'size' : 20})
plt.title('SubSample Correlation Matrix \n (use for reference)', fontsize = 14)

# We see that some of the variables are strongly correlated.

### Anomaly detection for improving classification accuracy 
## Removing outliers - SKIPPED for now 


### Dimensionality Reduction and Clustering - looking for patterns in the data 

# Separate features and classes
X = new_df.drop('Class', axis = 1)
y = new_df['Class']

# t-SNE implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components = 2, random_state = 42).fit_transform(X.values)
t1 = time.time()
print('T-SNE took {:.2} s'.format(t1-t0))

# PCA implementation
t0 = time.time()
X_reduced_PCA = PCA(n_components = 2, random_state= 42).fit_transform(X.values)
t1 = time.time()
print('PCA took {:.2} s'.format(t1-t0))

# TruncatedSVD implmentation 
t0 = time.time()
X_reduced_SVD = TruncatedSVD(n_components = 2, algorithm = 'randomized', random_state= 42).fit_transform(X.values)
t1 = time.time()
print('Truncated SVD took {:.2} s'.format(t1-t0))

# Plot the results of the dim. reduction algos 
f, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (24,6))
f.suptitle('Dimensionality reduction', fontsize = 14)

blue_patch = mpatches.Patch(color = '#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

# t-SNE scatter plot 
ax1.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=(y == 0), cmap = 'coolwarm', label = 'No Fraud', linewidths = 2)

ax1.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c = (y == 1),  cmap = 'coolwarm', label = 'Fraud', linewidths = 2)

ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles = [blue_patch, red_patch])

# TruncatedSVD scatter plot
ax2.scatter(X_reduced_PCA[:,0], X_reduced_PCA[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_PCA[:,0], X_reduced_PCA[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_SVD[:,0], X_reduced_SVD[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_SVD[:,0], X_reduced_SVD[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()

### Classification using *classifiers*

# Undersampling before cross validating (prone to overfit)
X = new_df.drop('Class', axis = 1)
y = new_df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Set a dictionary of classifiers to use 
classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

# Implement the classifiers using a for loop 
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv = 5)
    print('Classifiers: ', classifier.__class__.__name__, 'has a training score of', 
          round(training_score.mean(), 2) * 100, '% accuracy score')

### Optimize the parameters using GridSearchCV

## Logistic regression 

# List parameters and options to try
log_reg_params = {'penalty' : ['none', 'l2'], 'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Fit the GridSearchCV with the given parameters
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)

# Get the best results
log_reg = grid_log_reg.best_estimator_

## kNN

# List parameters and options to try
knn_params = {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# Fit the GridSearchCV with the given parameters
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params)
grid_knn.fit(X_train, y_train)

# Get the best results
knn = grid_knn.best_estimator_


## Support Vector Classifier

# List parameters and options to try
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

# Fit the GridSearchCV with the given parameters
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_

## Decision Trees

# List parameters and options to try
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}

# Fit the GridSearchCV with the given parameters
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# Decision trees best estimator
tree_clf = grid_tree.best_estimator_

## Get train CV score for all the models (with oversampling)
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knn, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')














