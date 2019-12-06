import pandas as pd
import numpy as np
import uuid

class PreDeCon:

    def __init__(self, e, m, l, d):
        self.e = e
        self.m = m
        self.l = l
        self.d = d

    def core(self, o):
        return True

    def neighbourhood(self, o):
        N = []
        # returns a list of indexes which are reachable in neighborhood
        return N

    def reachable(self, q):
        R = []
        # returns a list of indexes which are density reachable
        return R

    def fit(self, D):
        D = pd.DataFrame(data=D)
        for index, row in D.iterrows():
            if self.core(row):
                currentID = uuid.uuid4()
                queue = self.neighbourhood(row)
                while len(queue) != 0:
                    q = queue.pop(0)
                    R = self.reachable(q)
                    for x in R:
                        if D.iloc["label", x] == np.nan:
                            queue.append(x)
                        if D.iloc["label", x] == np.nan or D.iloc["label", x] == "noise":
                            D.iloc["label", x] = currentID
            else:
                row["label"] = "noise"
        return D
