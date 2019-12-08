import pandas as pd
import numpy as np
import uuid


class PreDeCon:
    """
    PreDeCon - "subspace PREference weighted DEnsity CONnected clustering"
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5825&rep=rep1&type=pdf
    """

    def __init__(self, e, m, l, d):
        """
        :e: Epsilon - The maximum distance that limits the epsilon-neighborhood of a point (real number)
        :m: Mu - The minimum number of points in an epsilon neighborhood for a point to be considered a core point (natural number)
        :l: Lambda - The maximum dimensionality of the searched clusters (natural number)
        :d: Delta - The maximum variance along one or more attributes (real number)
        """
        self.e = e
        self.m = m
        self.l = l
        self.d = d

    def preference_weights(self, row, indexes, D, k):
        df = D.iloc[indexes]
        N = df.shape[0]
        df = df.sub(row, axis='columns')
        df = df.apply(lambda x: (x ** 2) / N, axis=1)
        df = df.sum(axis=0)
        df = df.apply(lambda x: 1 if x > self.d else k)
        return df.values, df.value_counts()[k]

    def neighbourhood(self, row, D):
        df = pd.DataFrame(D.values - row.values, columns=D.columns)
        distances_pref = df.copy()
        df = df.apply(np.linalg.norm, axis=1)
        # the indexes of the e neighborhood
        idx = df[df <= self.e].index
        # get the preference weights
        w, pdim = self.preference_weights(row, idx, D, 100)
        # new weighted neighborhood
        distances_pref = distances_pref.apply(lambda x: ((x ** 2) * w).sum() ** .5, axis=1)
        idx = distances_pref[distances_pref <= self.e].index
        # returns a list of indexes which are reachable in preferred neighborhood
        return idx, pdim

    def reachable_getidx(self, row, D):
        """
        :row: The Element for which the density reachable elements should be computed
        :D: The entire data set
        Returns an index of all elements in D that are density reachable from the given element (=row).
        """
        df = pd.DataFrame(D.values - row.values, columns=D.columns)
        df = df.apply(np.linalg.norm, axis=1)
        idx = df[df <= self.e].index
        return idx

    def reachable(self, queue, D):
        df = D.copy()
        idx = queue.copy()
        for x in idx:
            idx_tmp = self.reachable_getidx(D.iloc[x], df)
            df.drop(idx_tmp)
            idx.append(idx_tmp)
        # returns a list of indexes which are density reachable
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


