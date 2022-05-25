import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

x = [1,5,1.5,8,1,9]
y = [2,8,1.8,8,0.6,11]
plt.scatter(x,y)
plt.show()
X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
Y = [0,1,0,1,0,1]
clasif = svm.SVC(kernel='linear',C=1.0)
clasif.fit(X,Y)
w = clasif.coef_[0]
print(w)
a = -w[0]/w[1]
xx = np.linspace(0,12)
yy = a*xx-clasif.intercept_[0]/w[1]
plt.plot(xx,yy,label='Linea de división')
plt.scatter(X[:,0],X[:,1],c=Y)
plt.legend()
plt.show()