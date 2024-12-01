import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if t[v].isna().any():
            if is_numeric_dtype(t[v]):
                t[v] = t[v].fillna(t[v].mean())
            else:
                # Same fix for non-numeric columns
                t[v] = t[v].fillna(t[v].mode()[0])


class acp():
    def __init__(self, t, variabile):
        assert isinstance(t, pd.DataFrame)
        self.__variabile = variabile
        self.__x = t[variabile].values

    @property
    def x(self):
        return self.__x

    def fit(self, std=True, nlib=0, procent_min=80):
        n, m = self.__x.shape
        x_ = self.__x - np.mean(self.__x, axis=0)
        if std:
            x_ = x_ / np.std(self.__x, axis=0)
        r_v = (1 / (n - nlib)) * x_.T @ x_
        valp, vecp = np.linalg.eig(r_v)
        # print(valp, vecp, sep="\n")
        k = np.flip(np.argsort(valp))
        # print(k)
        self.__alpha = valp[k]
        self.__a = vecp[:, k]
        self.__c = x_ @ self.__a
        alpha_ = np.cumsum(self.__alpha) * 100 / sum(self.__alpha)
        k1 = np.where(alpha_ > procent_min)[0][0] + 1
        if std:
            k2 = np.nan
        else:
            k2 = np.where(self.__alpha <= 1)[0][0]
        eps = self.__alpha[1:] - self.__alpha[:m - 1]
        sigma = eps[1:] - eps[:m - 2]
        negative = sigma < 0
        if any(negative):
            k3 = np.where(negative)[0][0] + 2
        else:
            k3 = np.nan
        self.__criterii = (k1, k2, k3)

    @property
    def alpha(self):
        return self.__alpha

    @property
    def a(self):
        return self.__a

    @property
    def c(self):
        return self.__c

    @property
    def criterii(self):
        return self.__criterii



