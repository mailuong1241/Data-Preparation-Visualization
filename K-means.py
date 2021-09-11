from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T
def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)

def init_centroids(X,k):
  centroids = X.copy()
  np.random.shuffle(centroids)
  return centroids[:k]
  while True:
    labels = closest_centroid(X,centroids)

    new_centroids = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

    if np.all(centroids==new_centroids):
      break
    centroids = new_centroids

  return centroids, labels

def closest_centroid(X, centroids):
    distances = np.sqrt(((X - centroids[:,np.newaxis])**2).sum(axis=2))
    return np.argmin(distances,axis = 0)
init_centroids(X,3)
