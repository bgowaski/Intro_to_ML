import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC # "Support vector classifier"

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#MAIN
input_file = 'features1.csv'
dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
X1 = df[df.columns[0]]
X2 = df[df.columns[1]]
X = list(df[[df.columns[1], df.columns[2]]].itertuples(index=False, name=None))
y = df.iloc[:,0].as_matrix()
X = np.array(X)

clf = SVC(kernel='rbf', C=1)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap='plasma')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=5, lw=1, facecolors='none');

plt.show()