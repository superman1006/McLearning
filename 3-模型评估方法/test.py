import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784",parser='auto')

X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# 洗牌操作
import numpy as np
# 将所有样本打乱
shuffle_index = np.random.permutation(60000)
# .iloc[]函数是pandas中根据索引提取数据的方法
X_train = X_train.iloc[shuffle_index]
y_train = y_train[shuffle_index]

y_train_5 = (y_train=="5")
y_test_5 = (y_test=="5")

from sklearn.linear_model import SGDClassifier
# SGDClassifier(stochastic gradient descent)是一个随机梯度下降分类器，适合处理大型数据集
sgd_clf = SGDClassifier(max_iter=5,random_state=42)
sgd_clf.fit(X_train,y_train_5)

sgd_clf.predict([X.iloc[35000]])



