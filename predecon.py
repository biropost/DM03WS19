import uuid

import numpy as np
import pandas as pd


class PreDeCon:
    """
    PreDeCon - "subspace PREference weighted DEnsity CONnected clustering"
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5825&rep=rep1&type=pdf
    """

    def __init__(self, e, m, l, d, k):
        """
        :e: Epsilon - The maximum distance that limits the epsilon-neighborhood of a point (real number)
        :m: Mu - The minimum number of points in an epsilon neighborhood for a point to be considered a core point (natural number)
        :l: Lambda - The maximum dimensionality of the searched clusters (natural number)
        :d: Delta - The threshold for small eigenvalues (real number)
        :k: Kappa - The penalty factor for deviations in preferred dimensions (real number)
        """
        self.e = e
        self.m = m
        self.l = l
        self.d = d
        self.k = k

    def preference_weights(self, row, indexes, D):
        df = D.iloc[indexes]
        N = df.shape[0]
        df = df.sub(row, axis='columns')
        df = df.apply(lambda x: (x ** 2) / N, axis='columns')
        df = df.sum(axis='rows')
        df = df.apply(lambda x: 1 if x > self.d else self.k)

        if self.k in df.value_counts():
            return df.values, df.value_counts()[self.k]

        return df.values, 0

    def neighbourhood(self, row, D):
        """
        :row: The Element for which the preferred-neighborhood-reachable elements should be computed
        :D: The entire data set
        Returns a list of indexes which are reachable in the preferred neighborhood
        """
        # the indexes of the e-neighborhood
        idx = self.reachable_getidx(row, D)
        # get the preference weights
        w, pdim = self.preference_weights(row, idx, D)
        # new weighted neighborhood
        distances_pref = pd.DataFrame(D.values - row.values, columns=D.columns)
        distances_pref = distances_pref.apply(lambda x: ((x ** 2) * w).sum() ** .5, axis='columns')
        idx = distances_pref[distances_pref <= self.e].index
        return idx, pdim

    def reachable_getidx(self, row, D):
        """
        :row: The Element for which the density reachable elements should be computed
        :D: The dataset that should be searched
        Returns an index of all elements in D that are density reachable from the given element (=row), ie. are in the e-neighborhood.
        """
        df = pd.DataFrame(D.values - row.values, columns=D.columns)
        df = df.apply(np.linalg.norm, axis='columns')
        return df[df <= self.e].index

    def reachable(self, queue, D):
        """
        :queue: Index of elements in D
        :D: The data set that should be searched
        Returns and index of points in D that are density reachable from any point in 'queue'
        """
        df = D.copy()
        idx = queue.copy()
        for x in idx:
            idx_tmp = self.reachable_getidx(D.iloc[x], df)
            df.drop(idx_tmp)
            idx.append(idx_tmp)
        return idx

    def fit(self, D):
        """
        :D: The dataset that should be processed
        Returns a vector where each point in the input data set is either assigned to a cluster or noise.
        Each cluster is identified by a unique id.
        """
        assignments = np.full((len(D), 1), np.nan, dtype=np.object)
        for index, row in D.iterrows():
            if pd.isnull(assignments[index]) or assignments[index] == "noise":
                queue, pdim = self.neighbourhood(row, D)
                if pdim <= self.l and len(queue) >= self.m:
                    current_id = uuid.uuid4()
                    while len(queue) != 0:
                        R = self.reachable(queue, D)
                        queue = np.delete(queue, 0)
                        for x in R:
                            if pd.isnull(assignments[x]):
                                np.append(queue, x)
                            if pd.isnull(assignments[x]) or assignments[x] == "noise":
                                assignments[x] = str(current_id)
                else:
                    assignments[index] = "noise"
        return assignments
