import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    # return 11 * np.sin(x) + 7 * np.cos(5 * x)
    return -np.square(x)


# 1.参数初始化
narvs = 1  # 变量个数
T0 = 100  # 初始温度
T = T0  # 迭代过程中温度会发生变化，第一次迭代的温度即为T0
maxgen = 200  # 最大迭代次数
Lk = 100  # 每个温度下的迭代次数
alpha = 0.95  # 温度的衰减系数
x_lb = -3  # x的下界
x_ub = 3  # x的上界
d = (x_ub - x_lb) / 100

# 2.随机生成一个初始解
x0 = np.random.uniform(x_lb, x_ub)
y0 = fun(x0)  # 计算当前解的函数值

# 3.定义一些中间变量，方便画图和输出结果
max_y = y0  # 初始化找到的最佳的解对应的函数值
best_x = x0
MAXY = np.zeros((maxgen, 1))  # 记录每一次外层循环后找到的最优解

# 4.模拟退火过程
for i in range(maxgen):
    for j in range(Lk):
        # 产生新解
        while True:
            # 不符合要求就一直循环
            x_new = x0 + np.random.uniform(-d, d) * T
            if x_lb <= x_new <= x_ub:
                break
        y_new = fun(x_new)
        # 如果新解更优秀就直接更新，不优秀以一定接受
        if y_new > y0:
            x0 = x_new
            y0 = y_new
        else:
            p = np.exp(-(y0 - y_new) / T)
            if np.random.random() < p:
                x0 = x_new
                y0 = y_new
        # 更新目前最优秀的解
        if y0 > max_y:
            max_y = y0
            best_x = x0
    MAXY[i] = max_y
    T = alpha * T  # 温度下降
    print("局部最优解：%.4f" % max_y)

print(max_y)
print(best_x)

x = np.linspace(x_lb, x_ub, 1000)
y = fun(x)
plt.plot(x, y)
plt.scatter(best_x, max_y,
            color="red", s=50,
            marker="*")
plt.show()
