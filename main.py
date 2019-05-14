import scipy.linalg as linalg
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import operator
import pandas as pd
import ssaCore

N = 600
L = 250 # Длинна гусеницы
K = N - L + 1 # Количество столбцов в траекторной матрице
t = np.arange(0,N)
trend = 0.08 * t
p1, p2 = 11, 8
periodic1 = 0.9 * np.sin(2*np.pi*t/p1)
periodic2 = 0.8 * np.sin(2*np.pi*(t+0.09)/p2)
noise = np.random.normal(0, 0.8, N)

F = trend + periodic1 + periodic2 + noise
ssa = ssaCore.SSA(F, L)
ssa.ssa()
ssa.getComponents()
ssa.filterComponents