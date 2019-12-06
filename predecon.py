import pandas as pd
import numpy as np
import uuid

class PreDeCon:

    def __init__(self, e, m, l, d):
        self.e = e
        self.m = m
        self.l = l
        self.d = d

    def preference_weights(self, row, indexes, D, k):
        df = D.iloc[indexes]
        N = df.shape[0]
        df.sub(row, axis='columns')
        df.apply(lambda x: (x**2)/N, axis=1)
        df.sum(axis=0)
        df.apply(lambda x: 1 if x > self.d else k)
        return df.values, df.value_counts()[k]

    def neighbourhood(self, row, D):
        df = D.sub(row, axis='columns')
        distances_pref = df.copy()
        df.apply(np.linalg.norm, axis=1)
        # the indexes of the e neighborhood
        idx = df[df >= self.e].index
        # get the preference weights
        w, pdim = self.preference_weights(row, idx, D, 100)
        # new weighted neighborhood
        distances_pref.apply(lambda x: (x**2*w).sum()**.5, axis=1)
        idx = distances_pref[distances_pref >= self.e].index
        # returns a list of indexes which are reachable in preferred neighborhood
        return idx, pdim

    def reachable_getidx(self, row, D):
        R = []
        df = D.sub(row, axis='columns')
        df.apply(np.linalg.norm, axis=1)
        idx = df[df <= self.e].index
        return idx

    def reachable(self, row, D):
        df = D.copy()
        idx = self.reachable_getidx(row, df)
        df.drop(idx)
        for x in idx:
            idx_tmp = self.reachable_getidx(row, df)
            if len(idx_tmp) > 0 :
                df.drop(idx_tmp)
                idx.append(idx_tmp)
        # returns a list of indexes which are density reachable
        return idx

    def fit(self, D):
        D = pd.DataFrame(data=D)
        for index, row in D.iterrows():
            if row["label"] == np.nan or row["label"] == "noise":
                queue, pdim = self.neighbourhood(row, D)
                if pdim <= self.l and len(queue) >= self.m:
                    currentID = uuid.uuid4()
                    while len(queue) != 0:
                        q = queue.pop(0)
                        R = self.reachable(q, D)
                        for x in R:
                            if D.iloc["label", x] == np.nan:
                                queue.append(x)
                            if D.iloc["label", x] == np.nan or D.iloc["label", x] == "noise":
                                D.iloc["label", x] = currentID
                else:
                    row["label"] = "noise"
        return D
