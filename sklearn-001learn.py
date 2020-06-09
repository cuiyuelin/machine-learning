import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets  # 数据集
from sklearn.model_selection import train_test_split  # 训练测试集拆分
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据归一化 均值方差归一化 最值归一化
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # KNN算法
# KNeighborsRegressor 解决K近邻回归问题
# 地址 http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
from sklearn.metrics import accuracy_score  # 分类准确度
from sklearn.model_selection import GridSearchCV  # 网格搜索  目的 寻找最好的超参数  CV交叉验证

# 从sklearn中导入数据集
# iris = datasets.load_iris()  # 数据集像一个字典 鸢尾花数据集
# print(iris.keys())
# iris.DESCR  数据集描述  包括多少样本  多少特征等
# iris.target 是 一个数组  样本的类型
# iris.target_names   样本的类型名称
# feature_names = iris.feature_names 特征名称
# X = iris.data[:, :2]
# print(X.shape)
# plt.scatter(X[:, 0], X[:, 1])
# y = iris.target
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="o")
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="+")
# plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", marker="x")
# plt.show()

# X = iris.data[:, 2:]
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="o")
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="+")
# plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", marker="x")
# plt.show()

# KNN  K近邻算法原理  解决分类问题  sklearn封装操作
# KNN比较特殊 因为没有模型  所以也把训练数据集本身叫做模型
# 计算新来的点到样本的距离 计算方法 采用欧拉距离
# from math import sqrt  # 求平方根
# distances = [sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
# nearest =np.argsort(distances)
# k = 6
# topK_y = [y_train[i] for i in nearest[:k]]
# from collections import Counter  # 数数
# votes = Counter(topK_y) # 返回一个字典
# predict_y = votes.most_common(1)[0][0] # 返回票数最多的1个元素 是一个列表 格式 [(1,5)]  所以通过 [0][0]取值

# 训练数据集  x_train 特征矩阵  y_train  标签向量
# 根据机器学习算法 生成模型的过程 叫做 fit(拟合)
# 输入样例根据模型 生成输出结果的过程 叫做 predict(预测)

# 使用sklearn封装操作KNN
# from sklearn.neighbors import KNeighborsClassifier  # 因为是面向对象的  所以要创建
# KNN_classifier = KNeighborsClassifier(n_neighbors=6)  # n_neighbors 就是K值
# KNN_classifier.fit(x_train, y_train)  # 返回自身self
# X_predict = x.reshape(1, -1)  # x 是一个输入样例  是一个数组  返回也是一个数组
# y_predict = KNN_classifier.predict(X_predict)

# 判断机器学习算法的性能
# 训练数据集 80%  测试数据集 20% =100%原始数据 tran_test_split
# 要保证取出数据的随机性
# 方法一 将X 和 y 整合 再打乱 shuffle 再split获取  X_train  y_train
# 方法一 生成一个 shuffle_indexes 长度 len(X) 再 按比例split  再根据索引值去获取  X_train  y_train
# 采用方法二
# shuffle_indexes = np.random.permutation(len(X))  # 0到 len(X)-1 的数随机排序  X 为原始数据矩阵
# test_ratio = 0.2
# test_size = int(len(X) * test_ratio)
# test_indexes = shuffle_indexes[:test_size]
# train_indexes = shuffle_indexes[test_size:]
#
# X_train = X[train_indexes]
# y_train = y[train_indexes]
# X_test = X[test_indexes]
# y_test = y[test_indexes]

# sklearn 中的 tran_test_split
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=666)

# sum(y_predict==y_test)/len(y_test) 返回测试数据和预测数据 的准确数

#  P25 分类的准确性
digits = datasets.load_digits()  # 数据集像一个字典 手写数据集
print(digits.keys())
# print(digits.DESCR)
X = digits.data
print(X.shape)
y = digits.target

# some_digit = X[666]
# print(y[666])
# some_digit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

my_knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)  # n_neighbors 就是K值
my_knn_clf.fit(X_train, y_train)  # 返回自身self
y_predit = my_knn_clf.predict(X_test)
print(accuracy_score(y_test, y_predit))  # 分类准确度
print(my_knn_clf.score(X_test, y_test))  # 分类准确度

#  P26 P27 超参数 和 网格搜索
#  (n_neighbors=3, weights='distance', p=2)  超参数
# KNN算法中没有模型参数
# 超参数 ： 在算法运行钱需要决定的参数  如何寻找好的超参数呢？
# 领域知识  经验数值  实验搜索(网格搜索)
param_grid = [{'weights': ['uniform'], 'n_neighbors': [i for i in range(1, 11)]},
              {'weights': ['distance'], 'n_neighbors': [i for i in range(1, 11)], 'p': [i for i in range(1, 6)]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)  # n_jobs 匹配四核 并行  verbose查看搜索的过程 值越大越详细
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)  # 这是一个 KNeighborsClassifier
print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_index_)

# 对距离的计算
# 欧拉距离 和 曼哈顿距离 可以总结成 明可夫斯基距离  超参数 P
# 更多的距离定义   定义与使用场景  不做具体分析
# 向量空间余弦相似度 Cosine Similarity
# 调整余弦相似度 Adjusted Cosine Similarity
# 皮尔森相关系数 Pearon Correlation Coefficient
# Jaccard相似系数 Jaccard Coefficient

# 模型参数 ： 算法过程中学习的参数

# P28 数据归一化 Feature Scaling  问题：比如 一共两个特征 一个特征的值太太，另一个则相对较小（原因单位不同），则最大的值影响了整体结果。
# 解决方案：
# 方法一 将所有的数据映射到 0~1之间 （最值归一化 normalization） 用于 分布有明显边界的特征
# 方法二 将所有数据归一到均值为0，方差为1 的分布中 （均值方差归一化 standardization）用于数据没有明显边界；有可能存在极端数据值

# P29 对 测试数据集如何归一化  和 sklearn中方法的使用
# 测试数据集归一化要使用（mean_train 和 std_train 训练数据的均值和方差） 即 (X_test-mean_train)/std_train
standardScaler = StandardScaler()
standardScaler.fit(X_train)
standardScaler.mean_  # 均值 有下划线的这种变量  不是操作员传进去的变量  而是根据传进去的变量从而生成的变量
standardScaler.scale_  # 方差 standardScaler.std_ 这个已经被弃用 改用了scale_
X_train = standardScaler.transform(X_train)  # 将训练数据 均值方差归一化
X_test_standard = standardScaler.transform(X_test)
my_knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)  # n_neighbors 就是K值
my_knn_clf.fit(X_train, y_train)  # 返回自身self
print(my_knn_clf.score(X_test_standard, y_test))  # 分类准确度
# minMaxScaler = MinMaxScaler();

# P29 更多的有关K近邻算法  总结
# 可以解决分类问题 而且是天然可以解决多分类问题
# 可以解决回归问题 就是预测房价 成绩等
# 最大的缺点 效率低下 优化 使用树结构 例如 KD-Tree Ball-Tree
# 最大的缺点 高度数据相关 如果k近邻 K为三 其中两个数据为错误数据  则出现错误
# 最大的缺点 预测的结果不具备可解释性
# 最大的缺点 维数灾难  看似相近的两个点 维数越多  距离越来越大  解决办法 降维


print("END")
