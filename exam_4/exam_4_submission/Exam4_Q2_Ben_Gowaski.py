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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy import interpolate
from sklearn.metrics import mean_squared_error
def plot_2d_separator(classifier, X, fill=False, ax=None, eps=None):
    if eps is None:
        eps = X.std() / 2.
    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    # no decision_function
    decision_values = classifier.predict_proba(X_grid)[:, 1]

    if ax is None:
        ax = plt.gca()
    print(decision_values)
    plt.contourf(xx, yy, decision_values.reshape(X1.shape), cmap=plt.cm.Paired, alpha=0.8)
    #ax.plot(decision_values.reshape(X1.shape),  linestyle='-', color='red', linewidth=3)
    # temp = decision_values.reshape(X1.shape) #300 represents number of points to make between T.min and T.max
    # xnew = np.linspace(temp.min(),temp.max(),300)
    # power_smooth = spline(temp,power,xnew)
    #plt.plot(xnew,power_smooth)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    plt.title('MLP Decision Boundary')
    plt.show()
    
#MAIN
input_file = 'MLP.csv'
dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
X = list(df[[df.columns[1], df.columns[2]]].itertuples(index=False, name=None))
y = df.iloc[:,0].as_matrix()
X = np.array(X)
# set k folds = 10 
k = 10
# Find the best params out of a set
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (1,1,1)],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
# Initial MLP for cross validation
mlp = MLPRegressor(max_iter=10)
# Let kfold split the data
print("K = %d \n" % k)
kf = KFold(n_splits=k)
kf.split(X)
for train_indices, test_indices in kf.split(X):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
mlp_clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=k)
mlp_clf.fit(X_train, y_train)
# All results
mlp_scores = []
means = mlp_clf.cv_results_['mean_test_score']
stds = mlp_clf.cv_results_['std_test_score']
y_true, y_pred = y_test , mlp_clf.predict(X_test)
# mse = mean_squared_error(y_true, y_pred)
print("\nMLP:")
print('Best MLP parameters found for K= ',k ,':\n', mlp_clf.best_params_)
print('Results on the MLP accuracy test set:')
print("MLP mean_test_score: ",means,'\n')
print("MLP std_test_score: ",stds,'\n')
#print("MLP accuracy score: ",mlp_scores,'\n')
#https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
#print(classification_report(y_true, y_pred))

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
mlp_best = MLPRegressor(hidden_layer_sizes=(50,50,50),
    activation='relu',
    solver='adam',
    alpha=0.05,
    learning_rate='constant',
    max_iter=50)

fig = plt.figure()
mlp_best.fit(X_train, y_train)
plt.plot(mlp_best.loss_curve_,color='blue',label='Training Set')
mlp_best.fit(X_test,y_test)
plt.plot(mlp_best.loss_curve_,color='red',label='Testing Set')
fig.suptitle('Loss of Train (blue) vs. Test (red) Set', fontsize=18)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('Loss', fontsize=16)

#pd.DataFrame(mlp_best.loss_curve_).plot(color='blue')

#pd.DataFrame(mlp_best.loss_curve_).plot(color='red')
plt.show()
# y_true, y_pred = y_test, mlp_best.predict(X_test)
# best_mse = mean_squared_error(y_true, y_pred)
# mlp_best_accuracy = accuracy_score(y_true, y_pred)
# print("MLP Best accuracy score: ",mlp_best_accuracy,'\n')
# print("MLP Mean Squared Error: ",best_mse,'\n')
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=5, cmap='plasma')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5, cmap='plasma',alpha=0.6)
plt.show()
plot_2d_separator(mlp_best, X, fill=False)

plt.show()