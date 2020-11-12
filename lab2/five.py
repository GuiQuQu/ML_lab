# 逻辑回归带正则项
# 测试不满足贝叶斯假设的数据

import numpy as np
from four import sigmod, generate_data, shrink, get_right_rate, plot


def main():
    train_size1 = 60
    train_size0 = 60
    _lambda = 0.1
    learning_rate = 0.1
    train_size = train_size1 + train_size0
    train_data1 = generate_data(train_size1, mu1=0.2, mu2=0.3, y=1)
    train_data0 = generate_data(train_size0, mu1=2, mu2=3, y=0)
    x_list = train_data1[0] + train_data0[0]
    y_list = train_data1[1] + train_data0[1]
    w = random_grad_correct(x_list, y_list, train_size, learning_rate, _lambda, 300, 1e-5)
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
    title = 'lambda=' + str(_lambda) + ',data_size=' + str(train_size)
    axis = [-1, 4, -2, 4]
    plot((data0x1, data0x2), (data1x1, data1x2), w, title, text, axis)
    pass


def no_naive_bayes():
    train_size1 = 60
    train_size0 = 60
    _lambda = 0.01
    learning_rate = 0.1
    train_size = train_size1 + train_size0
    train_data1 = generate_no_naive_bayes(train_size1, mu1=0.2, mu2=0.3, y=1)
    train_data0 = generate_no_naive_bayes(train_size0, mu1=2, mu2=3, y=0)
    x_list = train_data1[0] + train_data0[0]
    y_list = train_data1[1] + train_data0[1]
    w = random_grad_correct(x_list, y_list, train_size, learning_rate, _lambda, 600, 1e-5)
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
    title = 'no naive bayes:lambda=' + str(_lambda) + ',data_size=' + str(train_size)
    axis = [-1, 4, -2, 4]
    plot((data0x1, data0x2), (data1x1, data1x2), w, title, text, axis)


def generate_no_naive_bayes(data_size, mu1, mu2, y):
    res_x = []
    res_y = []
    mean = [mu1, mu2]
    one = np.mat(np.array([1, 2]))
    two = np.mat(np.array([2, 1]))
    cov = np.vstack([one, two])
    resx = np.mat(np.random.multivariate_normal(mean, cov, data_size))
    for i in range(resx.shape[0]):  # 对于行数，样本个数
        x = resx[i, :].T
        print(x)
        print()
        res_x.append(x)
    for i in range(data_size):
        res_y.append(y)
    return res_x, res_y


# 带正则项的似然函数
def likelihood_func(x_list, y_list, w, _lambda, data_size):
    res = 0
    for l in range(data_size):
        res = res + y_list[l] * w.T * x_list[l] - np.log(1 + np.exp(w.T * x_list[l]))
    res = res + _lambda * w.T * w  # 正则修正
    return res


def get_gradient(x_list, y_list, w, _lambda, data_size):
    res = []
    for i in range(x_list[0].shape[0]):  # 行  维度
        _w = 0
        for j in range(data_size):  # 样本个数
            gz = sigmod(w, x_list[j])
            print('x_list[', j, '][', i, '0]:', x_list[j][i, 0], 'y_list:', y_list[j], '     gz', gz)
            print(gz)
            _w = float(_w + x_list[j][i, 0] * (y_list[j] - gz) + _lambda * _w)  # 正则项在最后
        res.append(_w)
    return np.mat(res).T


def random_grad_correct(x_list, y_list, data_size, learning_rate, _lambda, k, precision):
    w = np.mat(np.zeros((len(x_list[0]) + 1, 1)))
    x_list = shrink(x_list)
    # 增加前置的1
    xx_list = []
    for x in x_list:
        xx_list.append(np.vstack((np.ones(1), x)))
    # 随机选择数据
    # 没有选择
    value0 = float('inf')
    value1 = likelihood_func(xx_list, y_list, w, _lambda, data_size)
    for i in range(k):
        g = get_gradient(xx_list, y_list, w, _lambda, data_size)
        w = w + learning_rate * g
        value0 = value1
        value1 = likelihood_func(xx_list, y_list, w, _lambda, data_size)
        if (value1 - value0) < 0:
            learning_rate = learning_rate * 0.5  # 走过了，降低学习率
            print('i+1', i + 1, learning_rate)
            print()
        if g.T * g <= precision:
            print(i + 1)
            break
    return w


if __name__ == '__main__':
    # main()
    no_naive_bayes()
