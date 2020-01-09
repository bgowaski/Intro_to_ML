import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed
from scipy import stats
from sklearn.datasets.samples_generator import make_circles
from sklearn.svm import SVC # "Support vector classifier"
import pandas as pd 

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


# MAIN
input_file = 'features1.csv'
dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
X1 = df[df.columns[0]]
X2 = df[df.columns[1]]
#X = [df[df.columns[0]],df[df.columns[1]]]
# X1 = df.loc[1:]
# X2 = df.loc[:2]
#print(X)
#print(X2)
X, y = make_circles(337, factor=.1, noise=.1)
X = list(df[[df.columns[0], df.columns[1]]].itertuples(index=False, name=None))
X = np.array(X)
print(X)
# df
# X1 = df.loc[[1]]
# X2 = df.loc[[2]]



#clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#plot_svc_decision_function(clf, plot_support=False);

#3D plot of circles
# def plot_3D(elev=30, azim=30, X=X, y=y):
#     ax = plt.subplot(projection='3d')
#     ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
#     ax.view_init(elev=elev, azim=azim)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('r')

# interact(plot_3D, elev=[-90, 90], azip=(-180, 180),
#          X=fixed(X), y=fixed(y));

clf = SVC(kernel='rbf', C=1)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none');

plt.show()