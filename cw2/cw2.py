from sklearn.cluster import KMeans
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np

mns = [(5, 3), (15, 4), (10, 8)]
scales = [(2, 1), (1, 1), (1, 2)]

params = zip(mns, scales)

clusters = []

for parset in params:
    dist_x = norm(loc=parset[0][0], scale=parset[1][0])
    dist_y = norm(loc=parset[0][1], scale=parset[1][1])
    cluster_x = dist_x.rvs(size=100)
    cluster_y = dist_y.rvs(size=100)
    cluster = zip(cluster_x, cluster_y)
    clusters.extend(cluster)
    
import csv
def planet_reader():
    with open('LidarDataX.xyz', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for ax, ay, az in reader:
            yield(ax, ay, az)

for p in planet_reader():
    print(p)
    
    
x, y, z = zip(*clusters)
plt.figure()
plt.scatter(x, y, z)
plt.title('Points scattering in 3D', fontsize=14)
plt.tight_layout()
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

clusterer = KMeans(n_clusters=3)

X = np.array(clusters)
y_pred = clusterer.fit_predict(X)

red = y_pred == 0
blue = y_pred == 1
cyan = y_pred == 2

plt.figure()
plt.scatter(X[red, 0], X[red, 1], c="r")
plt.scatter(X[blue, 0], X[blue, 1], c="b")
plt.scatter(X[cyan, 0], X[cyan, 1], c="c")