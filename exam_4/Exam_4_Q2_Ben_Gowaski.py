# Exam 4 Ben Gowaski
# Source for SVM: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC 
import seaborn as sns
import tensorflow as tf
import keras
import snips as snp
from subprocess import check_output
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# import data to score
def get_data(fn, n_columns=None):
    df = pd.read_csv(fn)
    # column 0 is label values
    labels = df[df.columns[0]]
    labels.to_csv('labels_MLP.csv')
    # columns 1 and 2 are the features
    features = df[df.columns[1:]]
    if n_columns is not None:
        features = features[features.columns[:n_columns]]
    print(features)
    features.to_csv('features_MLP.csv')
#MAIN
get_data('MLP.csv')
input_ = pd.read_csv('features_MLP.csv', index_col=0).values
output = pd.read_csv('labels_MLP.csv', index_col=0).values[:, 0]
# set k folds = 10 
k = 10
selector = np.random.rand(output.shape[0])
# delta = 1/k
delta = 1/k
bucket_selectors = []
for b in range(k):
    bucket_selector = [n for n in range(output.shape[0]) if b * delta < selector[n] <= (b + 1) * delta]
    bucket_selectors.append(bucket_selector)
input_buckets = np.array([input_[s].tolist() for s in bucket_selectors])
output_buckets = np.array([output[s].tolist() for s in bucket_selectors])
mlp_scores = []
mlp_scores_soft = []
# Get svm_scores for K = 1 through 10
for n in np.arange(k):
    test_X, test_Y = input_buckets[n], output_buckets[n]
    indices = np.arange(k)
    indices = np.delete(indices, n)
    train_X, train_Y = np.concatenate(input_buckets[indices]), np.concatenate(output_buckets[indices])
    # Get svm_scores based on Gaussian, rbf
    # Get mlp_scoes for K 
    # Adapted from https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset/downloads/Dataset_spine.csv/comments
    # https://www.kaggle.com/ahmethamzaemra/mlpclassifier-example
    clf = MLPClassifier(hidden_layer_sizes=(1,1,1),activation='logistic', max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
    clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    mlp_scores.append(accuracy_score(test_Y, pred_Y))
    # softplus
    clf1 = MLPClassifier(hidden_layer_sizes=(1,1,1),activation='tanh', max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
    clf1.fit(train_X, train_Y)
    pred_Y = clf1.predict(test_X)
    mlp_scores_soft.append(accuracy_score(test_Y, pred_Y))
# print("MLP SCORES")
# print(mlp_scores)
# print("MLP SOFT SCORES")
# print(mlp_scores_soft)

loss_values = clf.loss_curve_
print (loss_values)
plt.plot(loss_values)
plt.show()
loss_values1 = clf1.loss_curve_
print (loss_values1)
plt.plot(loss_values1)
plt.show()