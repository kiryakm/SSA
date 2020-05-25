import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def toTs(X):
    """
    Диагональное усреднение матрицы 
    и перевод во временной ряд

    X: 2D np.array - матрица для перевода в ряд

    Возвращает 1D np.array - временной ряд
    """
    Xrev = X[::-1]
    return np.array([Xrev.diagonal(i,0).mean() for i in range(-X.shape[0]+1, X.shape[1])])

def toImg(X):
    """
    Реконструкция изображения
    
    X: 2D np.array - матрица, которую необходимо преобразовать

    Возвращает 2D np.array с реконструированным изображением
    """
    Ly = X.shape[0]
    Ky = X.shape[1]
    Lx = X.shape[2]
    Kx = X.shape[3]        
    Ximg = np.zeros([Ly+Ky-1, Lx+Kx-1])
    
    for s in range(Ly):
        for l in range(s+1):
            Ximg[s] += toTs(X[l][s-l]) / (s+1)
        
    for s in range(Ly, Ky):
        for l in range(Ly):
            Ximg[s] += toTs(X[l][s-l]) / (Ly)
        
    for s in range(Ky, Ky+Ly-1):
        for l in range(s-Ky+1, Ly):
            Ximg[s] += toTs(X[l][s-l]) / (Ky+Ly-s-1)
    return Ximg

def normImg(img):
    """
    Нормализировать изображение.
    Переводит все значения входной 
    матрицы в промежуток от 0 до 255

    img: 2D np.array - матрица, которую необходимо преобразовать

    Возвращает 2D np.array с преобразованными значениями
    """
    return ((img - img.min()) * \
            (1 / (img.max() - img.min()) * 255))

def findTexture(img, tex, comps = [1]):
    """
    Выделить текстуру на изображении

    img: экземпляр класса SSA2D с изображением
    tex: экземпляр класса SSA2D с текстурой
    comps: list - список компонент для реконструкции изображения

    Возвращает 2D np.array с реконструированным изображением
    """
    if img.Ly != tex.Ly or img.Lx != tex.Lx:
        print("The window size should be the same.")
        return
    Xe = np.array([np.outer(tex.U[:,i], tex.V[:,i]) for i in comps])
    Ximg = Xe.sum(axis=0) @ img.X

    recImg = np.zeros([img.Ly, img.Ky, img.Lx, img.Kx])
    for i in range(img.Ly):
        for j in range(img.Ky):
            recImg[i][j] = Ximg[i*img.Lx:i*img.Lx+img.Lx, \
                                j*img.Kx:j*img.Kx+img.Kx]
    recImg = toImg(recImg)
    return normImg(recImg)

def getDistance(img1, img2, k = 5):    
    """
    Получить расстояние между изображениями

    img1: экземпляр класса SSA2D с первым изображением
    img2: экземпляр класса SSA2D со вторым изображением
    k: int - количество собственных векторов для определения дистанции

    Возвращает float дистанцию
    """
    s1 = img1.Ys
    s2 = img2.Ys
    
    Y = img1.Y + img2.Y
    _, s12, _ =  np.linalg.svd(Y)

    distance = s1[0:k] + s2[0:k] - s12[0:k]
    return np.sum(distance)

class SSA2D():
    """
    Класс, реализующий метод 2D-SSA
    """
    def __init__(self, img, Ly, Lx):
        """
        Инициализация класса

        img: 2D np.array - изображение для разложения
        Ly: int - размер окна по y
        Lx: int - размер окна по x
        """
        self.img = img
        self.h, self.w = img.shape
        self.Ly, self.Lx = Ly, Lx
        self.p = Ly * Lx
        self.q = (self.h-Ly+1) * (self.w-Lx+1)
        self.Ky, self.Kx = self.h-Ly+1, self.w-Lx+1    
        self.trajectMatrix()
        self.SVD()

    def trajectMatrix(self):
        """
        Создание траекторной матрицы
        """
        self.X = np.zeros((self.p, self.q)).astype(np.float32)
        for i in range(self.Ky):
            self.X[:,i*self.Kx: i*self.Kx + self.Kx] = np.array([self.img[i:i+self.Ly, j:j+self.Lx].flatten() 
                                    for j in range(self.Kx)]).T

    def SVD(self):  
        """
        SVD разложение
        """
        self.S = self.X @ self.X.T
        self.U, self.s, self.V = np.linalg.svd(self.S)
        self.V = self.V.T 

        self.Y = self.S / np.trace(self.S)
        _, self.Ys, _ =  np.linalg.svd(self.Y)

    def reconstruct(self, comp = np.arange(10)):
        """
        Реконструкция изображения

        comp: list - список компонент для реконструкции изображения
        """
        self.Xe = np.array([np.outer(self.U[:,i], self.V[:,i]) for i in comp])
        self.Ximg = self.Xe.sum(axis=0) @ self.X

        self.recImg = np.zeros([self.Ly, self.Ky, self.Lx, self.Kx])
        for i in range(self.Ly):
            for j in range(self.Ky):
                self.recImg[i][j] = self.Ximg[i*self.Lx:i*self.Lx+self.Lx, \
                                              j*self.Kx:j*self.Kx+self.Kx]
        self.recImg = toImg(self.recImg)
        self.recImg = normImg(self.recImg)

    def showContributions(self, hist = True, ln = True, n = 12):
        """
        Отображение графика с вкладом элементарных матриц в изображение

        hist: bool. True - построить гистограмму, False - построить график по точкам
        ln: bool. True - показать ln(λ), False - показать вклад в процентах
        n: int - количество элементарных матриц для отображения
        """
        if ln == True:
            cont = np.array([np.log(self.s[i]) / np.log(100) for i in range(n)])
            label = "ln(λ)"
        else:
            cont = np.array([self.s[i] / sum(self.s) for i in range(n)])
            label = "Percent"

        if hist == True:
            plt.bar(np.arange(len(cont)), cont, align="edge", label=label)
            plt.xticks(np.arange(len(cont)), np.arange(len(cont)))
        else:
            plt.plot(np.arange(len(cont)), cont, marker="o", label=label)

        plt.title("Contribution of elementary matrices")
        plt.xlabel("№ of matrix")
        plt.legend()
        plt.show()
    
    def showImage(self):
        """
        Показать восстановленное изображение
        """
        try:
            img = Image.fromarray(self.recImg)
        except AttributeError:
            print ("There is no reconstructed image. \nTry reconstruct() first.")
            return
        img.show()
    
    def saveImage(self, path = "reconstructedImage.png"):
        """
        Сохранить восстановленное изображение

        path: string - путь для сохранения
        """
        try:
            img = Image.fromarray(self.recImg)
        except AttributeError:
            print ("There is no reconstructed image. \nTry reconstruct() first.")
            return
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(path)