import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#matplotlib inline
X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(100,2)
np.append(X, X1)
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.show()
from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
centroids = Kmean.cluster_centers_
print(centroids)
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.scatter(centroids[0][0], centroids[0][1], c = 'g')
plt.scatter(centroids[1][0], centroids[1][1], c = 'r')
plt.show()
