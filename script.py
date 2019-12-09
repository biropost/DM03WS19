import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff

from predecon import PreDeCon

fName = 'data/iris.arff'
tData, meta = arff.loadarff(fName)

D = pd.DataFrame(data=tData)
true_labels = D["class"]
D = D.drop("class", axis=1)

# 1. Run PreDeCon

start = time.time()
pdc = PreDeCon(e=0.5, m=1, l=2, d=0.05)
clustered_labels = pdc.fit(D)
end = time.time()
print("Runtime:", end - start, "seconds")

# 2. Flatten the clustered labels (contains nested arrays with 1 element each, possibly fix in PreDeCon.fit())

clustered_labels = list(itertools.chain.from_iterable(clustered_labels))

# 3. Get all unique labels

unique_labels = np.unique(clustered_labels)

# 4. Assign a random color to each unique cluster id

# color_map is a dict: label name -> random color
color_map = dict(zip(unique_labels, np.random.rand(len(unique_labels), )))

# 5. map clustered_labels to colors list

colors = list(map(color_map.get, clustered_labels))

# 6. Plot all subspaces with colored points

names = meta.names()
names.remove("class")

plt_index = 1

# now add colors based on clustered_labels

for i, name in enumerate(names):
    for j in range(i+1, len(names)):
        x = D[names[i]]
        y = D[names[j]]
        # color = np.sqrt(x ** 2 + y ** 2)
        plt.subplot(3, 2, plt_index)
        plt_index = plt_index + 1
        plt.scatter(x, y, s=30, c=colors, marker="+")

plt.show()
