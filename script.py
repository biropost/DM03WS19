import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import metrics

from predecon import PreDeCon

print("Loading Dataset...")

fName = 'data/iris.arff'
tData, meta = arff.loadarff(fName)

D = pd.DataFrame(data=tData)
true_labels = D["class"]
D = D.drop("class", axis=1)

print("Clustering Data using PreDeCon...")

start = time.time()

pdc = PreDeCon(e=0.5, m=1, l=2, d=0.05, k=100)
clustered_labels = pdc.fit(D)

end = time.time()
print(" |- Runtime:", end - start, "seconds")

unique_labels = np.unique(clustered_labels)
print(" |- Number of Clusters:", len(unique_labels))

print("Preparing Data for Plots...")

# Flatten the clustered labels (contains nested arrays with 1 element each, possibly fix in PreDeCon.fit())
clustered_labels = list(itertools.chain.from_iterable(clustered_labels))

# Assign a random color to each unique cluster id
color_map = dict(zip(unique_labels, np.random.rand(len(unique_labels), )))

# map clustered_labels to colors list
colors = list(map(color_map.get, clustered_labels))

print("Generating Subspace Plots...")

names = meta.names()
names.remove("class")
plt_index = 1
for i, name in enumerate(names):
    for j in range(i+1, len(names)):
        x = D[names[i]]
        y = D[names[j]]
        plt.subplot(3, 2, plt_index)
        plt_index = plt_index + 1
        plt.scatter(x, y, s=30, c=colors, marker="+")

plt.show()

print("Performance Evaluation")

ari = metrics.adjusted_rand_score(true_labels, clustered_labels)
print(" |- ARI:", ari)

ami = metrics.adjusted_mutual_info_score(true_labels, clustered_labels)
print(" |- AMI:", ami)

nmi = metrics.normalized_mutual_info_score(true_labels, clustered_labels)
print(" |- NMI:", nmi)

mi = metrics.mutual_info_score(true_labels, clustered_labels)
print(" |- MI:", mi)

h = metrics.homogeneity_score(true_labels, clustered_labels)
print(" |- Homogeneity:", h)

c = metrics.completeness_score(true_labels, clustered_labels)
print(" |- Completeness:", c)

v = metrics.v_measure_score(true_labels, clustered_labels)
print(" |- V-Measure:", v)

fmi = metrics.fowlkes_mallows_score(true_labels, clustered_labels)
print(" |- FMI:", fmi)
