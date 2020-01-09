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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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
    if hasattr(model, "decision_function"):
        P = model.decision_function(xy)
    else:
        P = model.predict_proba(xy)[:, 1]
    P = P.reshape(X.shape)
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
#https://github.com/amueller/scipy_2015_sklearn_tutorial/blob/master/notebooks/figures/plot_mlp_decision_function.py
def plot_mlp_decision_function(classifier, X, fill=False, ax=None, eps=None):
    if eps is None:
        eps = X.std() / 2.
    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    try:
        decision_values = classifier.decision_function(X_grid)
        levels = [0]
        fill_levels = [decision_values.min(), 0, decision_values.max()]
    except AttributeError:
        # no decision_function
        decision_values = classifier.predict_proba(X_grid)[:, 1]
        levels = [.5]
        fill_levels = [0, .5, 1]

    if ax is None:
        ax = plt.gca()
    # if fill:
    #     ax.contourf(X1, X2, decision_values.reshape(X1.shape),
    #                 levels=fill_levels, colors=['blue', 'red'])
    # else:
    ax.contour(X1, X2, decision_values.reshape(X1.shape), levels=levels,
               colors="black")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

#https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

#MAIN
input_file = 'Q1_output.csv'
dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
X = list(df[[df.columns[1], df.columns[2]]].itertuples(index=False, name=None))
y = df.iloc[:,0].as_matrix()
X = np.array(X)
# set k folds = 10 
k = 10
svm_scores = []
mlp_scores = []
# Find the best params out of a set
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
# Initial MLP for cross validation
mlp = MLPClassifier(max_iter=500)
# Let kfold split the data
print("K = %d \n" % k)
kf = KFold(n_splits=k)
kf.split(X)
for train_indices, test_indices in kf.split(X):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
svm_clf = SVC(kernel='linear', C=k)
accuracy = cross_val_score(svm_clf, X_train, y_train, scoring='accuracy', cv = k).mean() * 100
svm_clf.fit(X_train, y_train)
#print(svm_clf.score(X_test, y_test))
# Get svm_scores based on Gaussian, rbf
svm_scores.append(svm_score(X_train,y_train,X_test,y_test, type='rbf', C=k))
# Find best hyperparameters for SVM of k folds
best_svm_params = svc_param_selection(X_train,y_train, k)
# Get MLP best params and scores for K 
mlp_clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=k)
mlp_clf.fit(X_train, y_train)
# All results
means = mlp_clf.cv_results_['mean_test_score']
stds = mlp_clf.cv_results_['std_test_score']
y_true, y_pred = y_test , mlp_clf.predict(X_test)
mlp_scores.append(accuracy_score(y_true, y_pred))
# Best parameter sets
print("\nSVM:")
print(cross_val_score(svm_clf, X_train, y_train, scoring='accuracy', cv = k))
print('Best SVM parameters found for K= ',k ,':\n', best_svm_params)
#print("Accuracy of SVM is: " , accuracy,'\n')
print("SVM accuracy score: ",svm_scores,'\n')
print("\nMLP:")
print('Best MLP parameters found for K= ',k ,':\n', mlp_clf.best_params_)
print('Results on the MLP accuracy test set:')
print("MLP mean_test_score: ",means,'\n')
print("MLP std_test_score: ",stds,'\n')
print("MLP accuracy score: ",mlp_scores,'\n')
#https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
print(classification_report(y_true, y_pred))

# Get whole data set to run against
dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
X = list(df[[df.columns[1], df.columns[2]]].itertuples(index=False, name=None))
y = df.iloc[:,0].as_matrix()
X = np.array(X)
#Spliting Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=True)

# Now that we have chosen optimal hyperparams:
# Run MLP with best params
mlp_best = MLPClassifier(hidden_layer_sizes=(50,50,50),
    activation='relu',
    solver='adam',
    alpha=0.05,
    learning_rate='adaptive',
    max_iter=500)
mlp_best.fit(X_train, y_train)
y_true, y_pred = y_test, mlp_best.predict(X_test)
mlp_best_accuracy = accuracy_score(y_true, y_pred)
print("MLP Best accuracy score: ",mlp_best_accuracy,'\n')
#Plot SVM
# Best SVM parameters found for K=  10 :
#  {'C': 10, 'gamma': 0.1}
svm_best = SVC(kernel='rbf', C=10, gamma=0.1)
svm_best.fit(X_train, y_train)
svm_best_accuracy = svm_score(X_train,y_train,X_test,y_test, type='rbf')
print("SVM Best accuracy score: ",svm_best_accuracy,'\n')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=5, cmap='plasma')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5, cmap='plasma',alpha=0.6)
plot_svc_decision_function(svm_best)
plt.scatter(svm_best.support_vectors_[:, 0], svm_best.support_vectors_[:, 1],
            s=1, lw=1, facecolors='none');
plt.title('SVM Decision Boundary')
plt.show()
#Plot MLP
# Code to plot the decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=5, cmap='plasma')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5, cmap='plasma',alpha=0.6)
plot_mlp_decision_function(mlp_best, X, fill=False)
plt.title('MLP Decision Boundary')
plt.show()