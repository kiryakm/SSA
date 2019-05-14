import numpy as np
import matplotlib.pyplot as plt
import os
import operator

def XtoTS(Xi):
    """
    Усредняем побочные диоганали элементарной матрицы 
    и переводим в временной ряд
    """
    Xrev = Xi[::-1]
    return np.array([Xrev.diagonal(i).mean() for i in range(-Xi.shape[0]+1, Xi.shape[1])])

def getRemainder(f, fRestored):
    """
    Посчитать абсолютный и относительный остаток
    средний остаток и макс. остаток
    """
    remainder = list()
    percent = list()
    for i in range(len(f)):
        a = f[i]
        b = fRestored[i]
        remainder.append(abs(a - b))
        if a != 0:
            percent.append(abs((a - b) / a) * 100)
            if percent[len(percent)-1] > 1000000:
                percent[len(percent)-1] = 0
    index, value = max(enumerate(remainder), key=operator.itemgetter(1))
    print("Absolute", value)
    index, value = max(enumerate(percent), key=operator.itemgetter(1))
    print("Relative", value)
    print("Average absolute",sum(remainder)/len(remainder))
    print("Average relative", sum(percent)/len(percent),"\n")

N = 200 
L = 90 # Длинна гусеницы
K = N - L + 1 # Количество столбцов в траекторной матрице
t = np.arange(0,N)

# Задаем времянной ряд

# trend = 0.08 * t
# p1, p2 = 11, 8
# periodic1 = 0.9 * np.sin(2*np.pi*t/p1)
# periodic2 = 0.8 * np.sin(2*np.pi*(t+0.09)/p2)

# t = np.arange(0,N)
# trend = 0.001 * (t - 100)**2
# p1, p2 = 20, 30
# periodic1 = 2 * np.sin(2*np.pi*t/p1)
# periodic2 = 0.75 * np.sin(2*np.pi*t/p2)

trend = 20 + 0.0001 * (t ** 2)
p1, p2 = 10, 3
periodic1 = 10 * np.sin(2*np.pi*t/p1)
periodic2 = 2 * np.cos(20*np.pi*(t-0.5)/p2)

mu = 0
sigma = 0.9
noise = 2 * np.random.normal(mu, sigma, N) 

np.random.seed(123) 
noise = 2 * (np.random.rand(N) - 0.5)

F = trend + periodic1 + periodic2 + noise

# Преобразуем ряд в траекторную матрицу
X = np.column_stack([F[i:i+L] for i in range(0,K)])

# Сингулярное разложение
U, Sigma, V = np.linalg.svd(X)
V = V.T 
# Высчитываем элементарные матрицы Xi
XElem = np.array([Sigma[i] * np.outer(U[:,i], V[:,i]) for i in range(0, L)])

d = np.linalg.matrix_rank(X) # The intrinsic dimensionality of the trajectory space.

FiFj = list()
w = np.array(list(np.arange(L)+1) + [L]*(K-L-1) + list(np.arange(L)+1)[::-1])

for i in range(len(XElem)):
    for j in range(len(XElem)):
        FiFj.append(w + XtoTS(XElem[i]) + XtoTS(XElem[j]))

Wcorr = np.identity(d)
for i in range(d):
    for j in range(i+1,d):
        FF = sum(w * XtoTS(XElem[i]) * XtoTS(XElem[j]))
        Fi = np.sqrt(sum(w * XtoTS(XElem[i]) * XtoTS(XElem[i])))
        Fj = np.sqrt(sum(w * XtoTS(XElem[j]) * XtoTS(XElem[j])))
        Wcorr[i,j] = FF / (Fi * Fj)
        Wcorr[j,i] = Wcorr[i,j]

ax = plt.imshow(Wcorr)
plt.xlabel(r"$\tilde{F}_i$")
plt.ylabel(r"$\tilde{F}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$W_{ij}$")
plt.clim(0,1)
plt.title("The W-Correlation Matrix for the Toy Time Series")
plt.show()


# Графики

# ln = list()
# for i in range(12):
#     ln.append(np.log(Sigma[i]) / np.log(100))
# ts = np.arange(0, len(ln))
# plt.title("ln(sigma)")
# plt.plot(ts, ln, marker='o')
# plt.show()

# plt.plot(t, F, label = "F")
# plt.plot(t, trend, label = "Trend")
# plt.title("Time series and trend")
# plt.legend()
# plt.show()

# plt.plot(t, periodic1, label = "Pereodic 1")
# plt.plot(t, periodic2, label = "Pereodic 2")
# plt.title("Pereodic part")
# plt.legend()
# plt.show()

# plt.title("Noise")
# plt.plot(t, noise, label = "Noise")
# plt.legend()
# plt.show()

# Fi = XtoTS(XElem[0]+XElem[2]+XElem[3]+XElem[4]+XElem[5])
# print("F")
# getRemainder(F, Fi)
# plt.plot(t, F, label = "Original F")
# plt.plot(t, Fi, label = "Restored F",  color = "red")
# plt.title("F")
# plt.legend()
# plt.show()

# Fi = XtoTS(XElem[0])
# print("Trend")
# getRemainder(trend, Fi)
# plt.plot(t, trend, label = "Original trend")
# plt.plot(t, Fi, label = "Restored trend")
# plt.title("Trend")
# plt.legend()
# plt.show()

# Fi = XtoTS(XElem[[2,3]].sum(axis=0))
# print("Pereodic 1")
# getRemainder(periodic1, Fi)
# plt.plot(t, periodic1, label = "Original periodic 1")
# plt.plot(t, Fi, label = "Restored periodic 1", linestyle='dashed', color = "red")
# plt.title("Periodic 1")
# plt.legend()
# plt.show()

# Fi = XtoTS(XElem[[4,5]].sum(axis=0))
# print("Pereodic 2")
# getRemainder(periodic2, Fi)
# plt.plot(t, periodic2, label = "Original periodic 2")
# plt.plot(t, Fi, label = "Restored periodic 2", linestyle='dashed', color = "red")
# plt.title("Periodic 2")
# plt.legend()
# plt.show()

# Fi = XtoTS(XElem[11])
# plt.plot(t, noise, label = "Original noise")
# plt.plot(t, Fi, label = "Restored noise")
# plt.title("Noise")
# plt.legend()
# plt.show()


