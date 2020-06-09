import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets  # 数据集
from sklearn.model_selection import train_test_split  # 训练测试集拆分
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据归一化 均值方差归一化 最值归一化
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # KNN算法
# KNeighborsRegressor 解决K近邻回归问题
from sklearn.metrics import accuracy_score  # 分类准确度
from sklearn.model_selection import GridSearchCV  # 网格搜索  目的 寻找最好的超参数  CV交叉验证

# from sklearn.linear_model import lin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # MSE 均方误差 MAE 平均绝对误差 R方

# 线性回归算法
# 样本特征只有一个的时候，成为简单线性回归  利用最小二乘法
# 多个样本特征 表示多元线性回归 利用多元线性回归的正规方程解 Normal Equation  样本越多 时间复杂度越高 解决办法 梯度下降法
# 解决回归问题  预测性问题
# 思想简单，视线容易
# 是许多强大的非线性模型的基础
# 结果具有很好的可解释性
# 蕴含机器学习中的很多重要思想

# 1、通过分析问题，确定问题的（损失函数或者效用函数） 统称目的函数；
# 2、通过最优的损失函数或者效用函数，获得机器学习的模型。
# 以上就是  近乎所有 参数学习算法都是这样的套路
# 例如 线性回归  多项式回归 逻辑回归  svm 以及SVM 神经网络  进而出现一个学科  最优化原理（里面有一个 凸优化 处理一些特殊的东西）
# 欧米伽（i=1到m）(y(i)-ax(i)-b)的平方  这是一个典型的最小二乘法的问题  最小化误差的平方
# 字母上面带一个^ 读 hat 表示预测值 y头上加^ 侧面加(i)  便是x(i)的预测值
# 字母上面带一个- 读 bar 表示当前向量的均值
# 求极值问题  就是求导结果为零的问题
X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]])
y = np.array([1, 2, 3, 4, 5])
# 向量化运算 使用.dot 向量的乘法

# 衡量线性回归法的指标  MSE 均方误差  RMSE 均方根误差 量纲不同 在sklearn中不存在   MAE 平均绝对误差
# RMSE 是要优于 MAE
# P36 评价回归算法 R Square R的平方在0，1之间 越靠近1越好 但是R的平方要是负数 说明不适合线性回归
# mse = mean_squared_error(y_test, y_predict)
# mae = mean_absolute_error(y_test, y_predict)
# rmse = sqrt(mse)
# r2 = 1 - mse / np.var(y_test)
# r2 = r2_score(x_test, y_test)
# 波士顿房产数据
boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]  #
y = y[y < 50.0]  # 去除房价的边界值
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# print(lin_reg.coef_)
# print(lin_reg.intercept_)
print(lin_reg.score(X_test, y_test))

knn_reg = KNeighborsRegressor(n_neighbors=5, p=1, weights='distance')
knn_reg.fit(X_train, y_train)
print(knn_reg.score(X_test, y_test))

# P40 线性回归的可解释性和更多思考
np.argsort(lin_reg.coef_)
print(boston.feature_names[np.argsort(lin_reg.coef_)])
# print(boston.DESCR)

print("END")
