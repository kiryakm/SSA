import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import operator

def ts(x):
    """
    Усредняем побочные диоганали элементарной матрицы 
    и переводим в временной ряд
    """
    xrev = x[::-1]
    return np.array([xrev.diagonal(i).mean() for i in range(-x.shape[0]+1, x.shape[1])])

def nans(dims):
    """
    nans((M,N,P,...)) is an M-by-N-by-P-by-... array of NaNs.
    :param dims: dimensions tuple
    :return: nans matrix
    """
    return np.nan * np.ones(dims)

class SSA:
    def __init__(self):
        self.components = []

    def ssa(self, F, L):
        self.F = F
        self.L = L
        self.N = len(F)
        self.K = self.N - self.L - 1

        self.X = np.column_stack([self.F[i:i+self.L] for i in range(0,self.K)])
        self.d = np.linalg.matrix_rank(self.X) 
        self.U, self.sigma, self.V = np.linalg.svd(self.X)
        self.V = self.V.T 
        self.Xe = np.array([self.sigma[i] * \
            np.outer(self.U[:,i], self.V[:,i]) for i in range(0, self.L)])

    def getComponents(self, p=0.8):
        for i in range(self.d):
            for j in range(i+1, self.d):
                stat = stats.pearsonr(ts(self.Xe[i]), ts(self.Xe[j]))[0]
                if stat >= p:
                    self.components.extend((i,j))
        if 0 not in self.components:
            self.components.append(0)
        if 1 not in self.components:
            self.components.append(1)

    def filterComponents(self, p=0.04):   
        for i, j in zip(self.components[0::2], self.components[1::2]):
            percent = (self.sigma[i] + self.sigma[i+1])\
                / sum(self. sigma)
            if percent < p:
                self.components.pop(self.components.index(i))
                self.components.pop(self.components.index(j))

    def restore(self):
        self.Xf = np.array([ts(self.Xe[i]) for i in self.components])
        self.Xf = sum(self.Xf)
        

    def predict(self, nForecast, e=None, maxIter=10000):
        if not e:
            e = 0.01 * (np.max(self.F) - np.min(self.F))
        Xmean = self.F.mean()
        x = self.F - Xmean
        self.Xfor = nans(nForecast)


        for i in range(nForecast):
            x = np.append(x, x[-1])
            yq = x[-1]
            y = yq + 2 * e
            nIter = maxIter
            while abs(y - yq) > e:
                yq = x[-1]
            
                self.ssa(x, self.L)
                xr = np.array([ts(self.Xe[i]) for i in self.components])
                xr = sum(xr)

                y = xr[-1]
                x[-1] = y
                nIter -= 1
                if nIter <= 0:
                    print('number of iterations exceeded')
                    break

            self.Xfor[i] = x[-1]
        self.Xfor = self.Xfor + Xmean