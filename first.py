import numpy as np
import matplotlib
import sklearn
import pandas
import random

# for _ in range(5):
#    print("Hello , Machine Learning!")

# 创建列表
# data = [i * 2 for i in range(100)]

# print(len(data))
# 前十个
# print(data[0:10])

# 0到1的随机数  生成十个
# L = [random.random() for i in range(10)]
# print(L)

'''numpy模块下 数组与矩阵相关的方法'''
# numpy数组 只支持一种类型
print(np.__version__)
nparr1 = np.array([i for i in range(10)])
# print(nparr1.dtype)
# 创建 值为int零的数组
nparr2 = np.zeros(10, dtype=int)
# 创建 值为int零的数组
nparr3 = np.zeros((3, 5), dtype=int)
nparr3 = np.zeros(shape=(3, 5))
# np.ones()
# 根据指定数值填充矩阵
np.full(shape=(3, 5), fill_value=666.0)
# 根据指定步长 生成数组  步长是可以为float的  python本带的i for i in range(0,20,1) 只能为整型
nparr4 = np.arange(0.20, 0.9)
# 根据指定范围 生成等差数组  左臂右臂
nparr5 = np.linspace(0.20, 2)
# 根据指定范围 生成随机向量或矩阵
nparr6 = np.random.randint(0, 20, size=10)
np.random.seed(111)  # 随机种子保证前后输出一直
nparr7 = np.random.randint(0, 20, size=(3, 5))
np.random.seed(111)
nparr7 = np.random.randint(0, 20, size=(3, 5))

# 生成随机向量或矩阵 float
nparr8 = np.random.random(10)
nparr9 = np.random.random((3, 5))
# print(nparr9)

# 生成正态分布的随机向量或矩阵 float  均值为 0 方差 为100
nparr8 = np.random.normal(0, 100, 10)
nparr9 = np.random.normal(0, 100, (3, 5))

# print(nparr8)
# print("nparr8多少维度=======" + str(nparr8.ndim))
# print(nparr9)
# print("nparr9多少维度=======" + str(nparr9.ndim))
# print("nparr9=======" + str(nparr9[2, 2]))

# 修改子矩阵 会影响原矩阵
subX = nparr9[2, 3]
# 加上.copy()  就不会影响了
subX = nparr9[2, 3].copy()

# reshape 不会影响原nparr8数组  -1 代表 行无限 或者列无限
# print(nparr8.reshape(2, 5))

# numpy矩阵的合并与分割
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])
# np.vstack(列表 * 高度、维度) 垂直方向累加矩阵 np.hstack() 水平方向累加矩阵  两者输入参数可以是一个矩阵 一个向量

A = np.arange(1, 17).reshape((4, 4))
# print("A=======", A)
x1, x2 = np.split(A, [2], axis=1)
# print(x1)
# print(x2)
# np.vsplit(A,[2])   np.hsplit(A,[2])

# numpy 的加减乘除运算 是在每个特征值上操作  而不是list中的数据加长
# np.abs(A)  # 求每个特征的绝对值
# np.sin(A) np.cos(A) 支持三角参数
# np.exp(A) e的特征次方
# np.power(3,A) 3的特征次方  同 3**A
# np.log(A)  # 以e为底 特征的自然对数
# print(np.log10(A))

# 高等数学对矩阵的乘法  第一个矩阵的列数（column）和第二个矩阵的行数（row）相同时才有意义
G = np.arange(4).reshape(2, 2)
K = np.full((2, 2), 10)
K[1, 1] = 9
L = G.dot(K)
# print("L=======", L)
L = L.T  # 转置
# print("L=======", L)

v1 = np.array([1, 2])
# print("v1=======", v1)
L + v1  # 高等数学中没有意义  但是代码中是有意义的 是这个向量v1 与 矩阵的每一行  做加法
L * v1  # 高等数学中没有意义  但是代码中是有意义的 是这个向量v1 与 矩阵的每一行  做乘法
v1.dot(L)  # 高等数学对矩阵的乘法
L.dot(v1)  # 高等数学对矩阵的乘法  因为v1 没有规定是行向量  还是列向量  numpy会自动转换
# print(np.tile(v1, (2, 2)))  # 这个元组 代表 行堆叠2次 列堆叠2次

# 矩阵的逆  inv  逆矩阵 要求矩阵是一个方阵
invL = np.linalg.inv(L)  # invL 是 L 的 矩阵的逆
# print(L.dot(invL))  # 结果为 从左上角到右下角的对角线 都为一的矩阵 单位矩阵
#  有的时候  不是方阵的矩阵  也要求矩阵的逆 叫做 伪逆的矩阵 numpy提供了新方法
pinvL = np.linalg.pinv(L)
# print(L.dot(pinvL))

# 聚合操作
L = np.random.random(100)
# print(np.sum(L))  # 所有特征之和 或者 可以使用  L.sum()
# print(np.max(L))  # np.percentile(L,q=100) 同max 意思是在L中找一个大于100%特征的数
# print(np.min(L))  # np.percentile(L,q=0) 同max 意思是在L中找一个大于0%特征的数
np.percentile(L, q=100)

X = np.arange(16).reshape(4, -1)
np.sum(X, axis=0)

np.prod(X)  # 矩阵各个特征相乘
np.prod(X + 1)  # X + 1 矩阵各个特征都要加一
np.mean(X)  # 平均值  数学期望(每次可能结果的概率乘以其结果的总和)
np.median(X)  # 中位数  # np.percentile(L,q=0) 同max 意思是在L中找一个大于50%特征的数

np.var(X)  # 方差  每个样本值与全体样本值的平均数之差的平方值的平均数
np.std(X)  # 标准差  标准差是方差的算术平方根  标准差能反映一个数据集的离散程度

# 索引 arg
x = np.random.normal(0, 1, size=1000000)
np.argmin(x)  # 最小值的索引位置
x = np.arange(16)
np.random.shuffle(x)  # 没有返回值 将 数组本身打乱
x.sort()  # 没有返回值 将 数组本身排序
x = np.sort(x)  # 有返回值 将 数组本身打乱
print(x)
np.partition(x, 3)  # 比3大的放右边  比3小的放左边  （未排序）
np.argsort(x)  # 展示索引数组

# 索引 Fancy Indexing  降微
ind = [3, 5, 8]
bbb = x[ind]
print(bbb)
ind = np.array([[0, 2], [1, 3]])
bbb = x[ind]
print(bbb)
X = x.reshape(4, -1)
print(X)

row = np.array([0, 1, 2])
col = np.array([1, 2, 3])
X[row, col]
X[0, col]
X[:2, col]
col = [True, False, True, True]
X[1:2, col]  # 这种比较重要
print(X > 3)
print(2 * x == 24 - 4 * x)
print(np.sum(X > 3))
print(np.count_nonzero(X > 3))
print(np.any(X > 3))  # 任意一个满足 返回True
print(np.all(X > 3))  # 全部满足 返回False
print(np.all(X > 3, axis=1))  # 全部满足 返回False
print(np.sum((X > 3) & (X < 10)))
print((X > 3) & (X < 10))
X[X[:3] % 3 == 0.:]  # 根据矩阵的第三列找能被3整除 的行

# Pandas（预处理） 用更好的的计算方法  但是 sklearn接收的确实numpy类型的
# 所以先用Pandas计算  之后再转换成numpy矩阵


