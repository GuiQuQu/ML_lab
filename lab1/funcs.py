import numpy as np
import math
import matplotlib.pyplot as plt
import sys


# 第一步：生成数据，加入高斯噪声
# x在0到2pi之间 然后加入标准正太分布的噪声noise


# 以sin(2x)为实验函数
def generate_data(data_size, low=0, high=2 * np.pi):
    # x0 = np.linspace(low, high, num=data_size, endpoint=True)  # 生成从low到high的随机数，个数为data_szie个
    x0 = np.random.random(data_size) * (high - low)
    y0 = np.sin(2 * x0)
    # 加入标准高斯噪声
    # randn生成符合标准高斯分布的随机数
    noise = np.random.randn(data_size) * 0.1
    y = y0 + noise
    return x0, y


# 生成X——，x1^0，x1^1,x2^2....,x1^m
# 生成范德蒙行列式的转置
def vandermonde(x0, m):
    x = np.ones((x0.shape[0], 1), dtype=float)  # 范德蒙行列式转置的的第一列
    x1 = np.array(x0).reshape(x.shape)
    for i in range(1, m):
        # 按水平方向堆叠向量
        x = np.hstack((x, x1 ** i))  # 一列数据
    return x


# 解析法求解loss的最优解，无正则项
# w=(x转置*X)^(-1)*x转置*t  x为范德蒙行列式转置形式，t为带有噪音的观测数据
def best_answer_without_correct(x0, t, m):
    x = vandermonde(x0, m)
    res = np.linalg.inv(x.T @ x) @ x.T @ t.reshape(t.shape[0], 1)
    return res


# 解析法求解loss的最优解，有正则项
# w=(x转置*x+lambda*E)^(-1)*x转置*t
def best_answer_with_correct(x0, t, m, l_lambda):
    x = vandermonde(x0, m)
    return np.linalg.inv(x.T @ x + l_lambda * np.eye(m, m)) @ x.T @ t


# Ax=b
# 计算左右的差距,采用二范数 (Ax-b).T @ (Ax-b)
def get_loss(A, x, b):
    temp = A @ x
    return 0.5 * (temp - b).T @ (temp - b)


def conjugate_down(w0, A, b):
    # A = x.T @ x + l_lambda * np.eye((x.T @ x).shape)
    # b = x.T @ t
    # 共轭梯度法最多迭代A.shape[1]步就可以出结果
    k = A.shape[1]
    w = np.mat(w0)  # 初值
    r = np.mat(b - A @ w)  # 梯度反方向
    p = r  # 第一次选定的下降方向
    flag = 0
    loss = [get_loss(A, w0, b), float('inf')]
    precision = 1e-8
    while (math.fabs(loss[0] - loss[1]) >= precision) and num < k:
        t = float((p.T @ r) / (p.T @ A @ p))
        w = w + t * p
        r_perv = r
        r = b - A @ w
        if all(np.fabs(r) <= precision): break
        beta = -float((r.T @ r) / (r_perv.T @ r_perv))
        p = r + beta * p
        flag = (flag + 1) % 2
        loss[flag] = get_loss(A, w, b)
    return w


# w 计算得出的系数，是一个向量,其中有m个元素
# x = vandermonde(x0, m)
# return 计算的函数值
def get_func(w, x):
    res = x @ w
    return res


# 计算损失函数的值
# x vandermonde(x0, m)
# w 计算得出的参数
# t 带噪音的观测数据
def get_Ew(x, w, t):
    y = x @ w
    return 0.5 * (y - t).T @ (y - t)


def get_RMS(Ew, n):
    return np.sqrt(2 * Ew / n)


# 画出对比图像
# ori 原始数据
# m  阶数+1
# n 数据个数
# w 计算得出w
# method_name 方法名
def match_plot(ori, m, n, w, line_name, xy_space=[0, np.pi * 2, -2, 2]):
    title = 'm=' + str(m - 1) + ',train_data=' + str(n)
    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.axis(xy_space)
    # sin(2x)
    x = np.linspace(0, np.pi * 2, num=300)
    plt.plot(x, np.sin(2 * x), color='y', label='sin(2x)')
    sol_y = get_func(w, vandermonde(x, m))
    plt.plot(x, sol_y, label=line_name)
    plt.scatter(ori[0], ori[1], color='r', label='training data')
    plt.legend()
    plt.show()
# m = 10
# n = 20
# # 生成数据
# ori = generate_data(n)
#
# # 求解
# w = best_answer_without_correct(ori[0], ori[1], m)
# # print('w',w)
# # 计算  拟合曲线的计算值
# y = get_func(w, vandermonde(ori[0], m))
# print('y:', y)
# # 画图
# # 拟合曲线
# pl.plot(ori[0], y)
# # 数据散点
# pl.scatter(ori[0], ori[1])
# pl.show()
