# 逻辑回归不带正则项
# 测试uci data
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from readcsv import read_csv


def main():
    train_size1 = 60
    train_size0 = 60
    train_size = train_size1 + train_size0
    test_size = 10
    train_data1 = generate_data(train_size1, mu1=0.2, mu2=0.3, y=1)
    train_data0 = generate_data(train_size0, mu1=2, mu2=3, y=0)
    x_list = train_data1[0] + train_data0[0]
    y_list = train_data1[1] + train_data0[1]
    w = random_grad_down(x_list, y_list, train_size, 0.1, 300, 1e-5)
    text = 'right rate=' + str(get_right_rate(x_list, y_list, w, train_size))
    print(text)
    # 画图
    data0x1 = []
    data0x2 = []
    data1x1 = []
    data1x2 = []
    for data in train_data0[0]:
        data0x1.append(float(data[0, 0]))
        data0x2.append(float(data[1, 0]))
    for data in train_data1[0]:
        data1x1.append(float(data[0, 0]))
        data1x2.append(float(data[1, 0]))
    title = 'data_size=' + str(train_size)
    axis = [-1, 4, -2, 4]
    plot((data0x1, data0x2), (data1x1, data1x2), w, title, text, axis)


def uci_data():
    file_name = 'divorce.csv'
    x_list, y_list = read_csv(file_name)
    data_size = len(y_list)
    w = random_grad_down(x_list, y_list, data_size, 0.1, 200, 1e-5)
    print('right rate=', get_right_rate(x_list, y_list, w, data_size))


def get_right_rate(x_list, y_list, w, data_size):
    # 计算准确率
    right_count = 0
    for i in range(data_size):
        new_x = np.vstack((np.ones(1), x_list[i]))
        pre_res = w.T * new_x
        if pre_res > 0 and y_list[i] == 1:
            right_count = right_count + 1
        if pre_res < 0 and y_list[i] == 0:
            right_count = right_count + 1
    # print('预测正确率 =', right_count / data_size)
    return right_count / data_size


# data0 标签为0的数据  (list[x1],list[x2])
# data1 标签为1的数据  (list[x1],list[x2])
# w (w0,w1,w2....).T  matrix
def plot(data0, data1, w, title, text, axis):
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(data1[0], data1[1], color='r', label='positive', marker='^')
    plt.scatter(data0[0], data0[1], color='g', label='negative', marker='o')
    # 画分类线
    x1 = np.linspace(-2, 5, endpoint=True, num=300)
    x2 = []
    for x in x1:
        x2.append(np.float(-w[1, 0] / w[2, 0] * x - w[0, 0] / w[2, 0]))
    plt.plot(x1, x2, label='classification')
    plt.legend()
    plt.text(2, -1.5, text, ha='left', rotation=0, wrap=True, color='b')
    plt.axis(axis)
    plt.show()


# data_size 数据个数
# mu1 x1正态分布均值
# mu2 x2正态分布均值
# y 类别
def generate_data(data_size, mu1, mu2, y):
    res_x = []
    res_y = []
    for i in range(data_size):
        x1 = 0.5 * np.random.randn(1) + mu1
        x2 = 0.5 * np.random.randn(1) + mu2
        res_x.append(np.mat(np.array([x1, x2])))
        res_y.append(y)
    return res_x, res_y


def grad(x_list, y_list, w, data_size):
    res = []
    for i in range(x_list[0].shape[0]):  # 行  维度
        _w = 0
        for j in range(data_size):  # 样本个数
            gz = sigmod(w, x_list[j])

            print('x_list[j][i,0] ', x_list[j][i, 0], 'y_list', y_list[j], '     gz', gz)
            print(gz)
            _w = float(_w + x_list[j][i, 0] * (y_list[j] - gz))
        res.append(_w)
    return np.mat(res).T


# x (1,x1,x2).T   matrix
# w (w0,w1,w2).T  matrix
def sigmod(w, x):
    fenmu = 1 + math.exp(-w.T * x)
    # print(fenmu)
    return 1 / fenmu
    pass


def likelihood_function(x_list, y_list, w, data_size):
    res = 0
    for l in range(data_size):
        res = res + y_list[l] * w.T * x_list[l] - np.log(1 + np.exp(w.T * x_list[l]))
    return res


# k 迭代次数
#
def random_grad_down(x_list, y_list, data_size, learning_rate, k, precision):
    w = np.mat(np.zeros((len(x_list[0]) + 1, 1)))
    x_list = shrink(x_list)
    # 增加前置的1
    xx_list = []
    for x in x_list:
        xx_list.append(np.vstack((np.ones(1), x)))
    # 随机选择50%数据
    choice_x_list = []
    select_size = int(data_size * 1)
    # choice_s = random.sample(range(data_size), select_size)
    # for choice in choice_s:
    #     choice_x_list.append(xx_list[choice])
    # xx_list = choice_x_list
    value0 = float('inf')
    value1 = likelihood_function(xx_list, y_list, w, select_size)
    for i in range(k):
        g = grad(xx_list, y_list, w, select_size)
        w = w + learning_rate * g

        value0 = value1
        value1 = likelihood_function(xx_list, y_list, w, select_size)
        if (value1 - value0) < 0:
            learning_rate = learning_rate * 0.5  # 走过了，降低学习率
            print('i+1', i + 1, learning_rate)
            print()
        if g.T * g <= precision:
            print(i + 1)
            break
    return w


def shrink(x_list):
    x_mat = x_list[0]
    res = []
    i = 1
    while i < len(x_list):
        x_mat = np.hstack((x_mat, x_list[i]))
        i = i + 1
    for i in range(x_mat.shape[0]):  # 行
        max = x_mat[i, 0]
        min = x_mat[i, 0]
        for m in range(x_mat.shape[1]):  # 列
            if max < x_mat[i, m]: max = x_mat[i, m]
            if min > x_mat[i, m]: min = x_mat[i, m]
        for j in range(x_mat.shape[1]):  # 列
            x_mat[i, j] = (x_mat[i, j] - min) / (max - min)
    for j in range(x_mat.shape[1]):
        res.append(x_mat[:, j])
    return res


if __name__ == '__main__':
    # uci_data()
    main()
