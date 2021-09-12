from __future__ import print_function 
import numpy as np
from copy import deepcopy
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
centroids=init_centroids(X,3)

plt.scatter(X0[:, 0], X0[:, 1], c ='b', marker = '.', s = 50)
plt.scatter(X1[:, 0], X1[:, 1], c = 'g', marker = '^', s = 50)
plt.scatter(X2[:, 0], X2[:, 1], c = 'r', marker = 'o', s =50)
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s = 100, c='k')



def closest_centroid(X, centroids):
    D = cdist(X,centroids)
    D2 = centroids[0] - X
    D3 = centroids[1] - X
    D4 = centroids[2] - X
    clusters = np.argmin(D,axis = 1)
    return clusters

closest_centroid(X,centroids)

k = 3
centroids_old = np.zeros(centroids.shape) 
centroids_update = deepcopy(centroids)
X.shape

error = np.linalg.norm(centroids_update - centroids_old)

while error != 0:
    centroids_old = deepcopy(centroids_update)
    
    for i in range(k):
        centroids_update[i] = np.mean(X[clusters == i], axis = 0)
    error = np.linalg.norm(centroids_update - centroids_old)
centroids_update
    

plt.scatter(centroids_update[:,0], centroids_update[:,1], marker = '*', c ='y', s=100)
