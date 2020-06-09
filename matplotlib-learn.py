import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 数据可视化工具
x = np.linspace(0, 10, 100)

# 折线图
y = np.sin(x)
cosy = np.cos(x)
# plt.plot(x, y)
# plt.plot(x, y, color="red", linestyle="--", label="sin(x)")
# plt.plot(x, cosy, color="green", label="cos(x)")
# plt.xlim(-5, 15)
# plt.ylim(0, 1.5)
# plt.axis([-5, 15, 0, 1.1])
# plt.xlabel("heihei")
# plt.ylabel("haha")
# plt.legend()  # plot中的参数label 不生效 所以添加的 意思是  添加图释
# plt.title("shabi")
# plt.show()

# 散点图  Scatter Plot
x = np.random.normal(0, 1, 10000)
y = np.random.normal(0, 1, 10000)
plt.scatter(x, y, alpha=0.1)  # alpha 透明度
plt.show()

# 例子

