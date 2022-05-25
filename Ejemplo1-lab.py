import xlrd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def cargar_datos(cols = ['Precio actual','Precio final']):
    data = pd.read_excel('Data10.xlsx')
    data = data[:100]

    x = np.array(data[cols].values)
    y = (data['Estado'].replace('Bajo', 0).replace('Alto', 1).values.tolist())
    return x, y

def proceso():
    x, y = cargar_datos()
    clasif = svm.SVC(kernel='linear', C=1.0)
    clasif.fit(x, y)
    w = clasif.coef_[0]
    a = -w[0] / w[1]
    print(w)
    a = -w[0] / w[1]
    xx = np.linspace(min(x[:,0]), max(x[:,0]))
    yy = a * xx - clasif.intercept_[0] / w[1]
    plt.plot(xx, yy, label='Linea de división')
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.xlabel('Precio final : ')
    plt.ylabel('Precio Actual : ')
    plt.legend()
    plt.show()

proceso()