#Exam 3 Question 2 Ben Gowaski
# derived from example code: https://gist.github.com/bthirion/1539342/25d7a45e46f7c517c46105ab413647bc3fb6b2e8
import pylab as pl
import numpy as np
from sklearn.mixture import GaussianMixture

# Generate 100 Samples
N = 100
np.random.seed(0)
meanVectors = np.array([[0.25, 0.75],
                    [0.75, 0.75],
                    [0.75, 0.25]])
covars = 0.1
X = np.random.normal(meanVectors, covars, size=(N, 3, 2))
X = X.reshape(-1, 2)
colors = (np.ones((N,1)) * np.arange(3)).reshape(-1)

pl.figure()
pl.scatter(X[:, 0], X[:, 1], c=colors, s=16, lw=0)
pl.title('input data')

n_components = np.arange(1, 16)
BIC = np.zeros(n_components.shape)

for i, n in enumerate(n_components):
    clf = GaussianMixture(n_components=n,
              covariance_type='diag')
    clf.fit(X)

    BIC[i] = clf.bic(X)

pl.figure()
pl.bar(n_components, BIC, label='BIC')
pl.legend(loc=0)
pl.xlabel('n_components')
pl.ylabel('BIC')

i_n = np.argmin(BIC)

clf = GaussianMixture(n_components[i_n])
clf.fit(X)
label = clf.predict(X)

pl.figure()
pl.scatter(X[:, 0], X[:, 1], c=label, s=16, lw=0)
pl.title('classification at min(BIC)')
pl.show()