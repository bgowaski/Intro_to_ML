# Exam 4 Ben Gowaski
# Source for SVM: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC 
import seaborn as sns
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
    labels.to_csv('labels.csv')
    # columns 1 and 2 are the features
    features = df[df.columns[1:]]
    if n_columns is not None:
        features = features[features.columns[:n_columns]]
    print(features)
    features.to_csv('features.csv')
# function to get score of the SVM
def svm_score(train_X, train_Y, test_X, test_Y, type='rbf', C=1):
    model = SVC(kernel=type, C=C)
    model.fit(np.array(train_X.tolist()), np.array(train_Y.tolist()))
    return model.score(test_X, test_Y)

def plot_svc_decision_function(model, plot_support=True):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[ 0], alpha=1,
               linestyles=['-'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#MAIN
# Convert data to get svm_scores
get_data('Q1_output.csv')
input_ = pd.read_csv('features.csv', index_col=0).values
output = pd.read_csv('labels.csv', index_col=0).values[:, 0]
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
svm_scores = []
mlp_scores = []

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
# Get svm_scores for K = 1 through 10
for n in np.arange(k):
    test_X, test_Y = input_buckets[n], output_buckets[n]
    indices = np.arange(k)
    indices = np.delete(indices, n)
    train_X, train_Y = np.concatenate(input_buckets[indices]), np.concatenate(output_buckets[indices])
    # Get svm_scores based on Gaussian, rbf
    svm_scores.append(svm_score(train_X, train_Y, test_X, test_Y, type='rbf', C=1))
    # Get mlp_scoes for K 
    # Adapted from https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset/downloads/Dataset_spine.csv/comments
    # https://www.kaggle.com/ahmethamzaemra/mlpclassifier-example
    clf = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=100, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
    clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    mlp_scores.append(accuracy_score(test_Y, pred_Y))
print("SVM SCORES")
print(svm_scores)
print("MLP SCORES")
print(mlp_scores)
#Plot MLP
loss_values = clf.loss_curve_
print (loss_values)
plt.plot(loss_values)
plt.show()
# Code to plot the decision boundary
input_file = 'Q1_output.csv'
dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
X = list(df[[df.columns[1], df.columns[2]]].itertuples(index=False, name=None))
y = df.iloc[:,0].as_matrix()
X = np.array(X)

clf = SVC(kernel='rbf', C=1)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap='plasma')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=1, lw=1, facecolors='none');
plt.show()