# 共轭梯度法求解多项式曲线拟合

import numpy as np
import matplotlib.pyplot as plt

from lab.funcs import \
    vandermonde, get_loss, \
    get_func, get_RMS, get_Ew


def main():
    m = 14
    n = 50
    test_size = 40
    # 产生数据
    ori_x, ori_y = generate_data(n)
    test_x, test_y = generate_data(test_size)
    # 确定梯度下降初始化内容，不带正则项
    w0 = np.zeros((m, 1))
    x = vandermonde(ori_x, m)
    A = x.T @ x
    b = x.T @ np.mat(ori_y).T
    # 共轭梯度下降
    w = conjugate_down(w0, A, b)
    # 画图
    match_plot((ori_x, ori_y), m, n, w)
    # 计算E_RMS
    _x = vandermonde(test_x, m)
    e_rms = get_RMS(get_Ew(_x, w, np.mat(test_y).T), test_size)
    print('without correct in test data:e_rms=', e_rms)
    # 带正则项
    la = np.exp(-9)
    A = x.T @ x + la * np.mat(np.eye(A.shape[0], A.shape[1]))
    w = conjugate_down(w0, A, b)
    # 计算E_RMS
    e_rms = get_RMS(get_Ew(_x, w, np.mat(test_y).T), test_size)
    print('with correct in test data:e_rms=', e_rms)
    # 画图
    match_plot((ori_x, ori_y), m, n, w, line_name='conjugate_down_with_correct')
    # 确定最好的lambda
    la_list = np.array([np.exp(i) for i in range(-15, 0)])
    e_rms_train = []
    e_rms_test = []
    for _la in la_list:
        A = x.T @ x + _la * np.mat(np.eye(A.shape[0], A.shape[1]))
        w = conjugate_down(w0, A, b)
        # 计算E_RMS
        e_rms_te = get_RMS(get_Ew(_x, w, np.mat(test_y).T), test_size)
        e_rms_test.append(float(e_rms_te))
        e_rms_tr = get_RMS(get_Ew(x, w, np.mat(ori_y).T), n)
        e_rms_train.append(float(e_rms_tr))
    # 画图 e_rms ---- lambda
    # plt.xlabel('lambda($e^{-x}$)')
    s = 'train_size=' + str(n) + ',test_size=' + str(test_size)
    plt.title(s)
    plt.xlabel('lambda')
    plt.ylabel('$E_{RMS}$')
    plt.xscale('log')
    plt.xticks([np.exp(-i) for i in range(5, 15, 2)], ['$e^{' + str(-i) + '}$' for i in range(5, 15, 2)])
    plt.plot(la_list, e_rms_train, color='r', label='train')
    plt.plot(la_list, e_rms_test, color='b', label='test')
    plt.legend()
    plt.show()


# 产生数据(参考函数为sin(2pix))
def generate_data(data_size, low=0, high=1):
    x0 = np.random.random(data_size) * (high - low)
    y0 = np.sin(2 * np.pi * x0)
    # 加入标准高斯噪声
    # randn生成符合标准高斯分布的随机数
    noise = np.random.randn(data_size) * 0.2
    y = y0 + noise
    return x0, y


def conjugate_down(w0, A, b):
    w = np.mat(w0)
    A = np.mat(A)
    b = np.mat(b)
    r = np.mat(b - A * w)
    p = r
    precision = 1e-5
    for i in range(A.shape[1] + 1):
        alpha = float((p.T @ r) / (p.T @ A @ p))  # alpha(k)
        w = w + alpha * p  # w(k+1)= alpha(k)*p(k)
        r_prev = r  # 记录r(k-1)
        r = r - alpha * A * p  # r(k+1)=r(k)-alpha(k)*p(k)
        if r.T @ r < precision: break  # 精度要求
        beta = float((r.T @ r) / (r_prev.T @ r_prev))  # beta(k) =[r(k+1).T @ r(k+1)]/[r(k).T @ r(k)]
        p = r + beta * p  # p(k+1)=r(k+1)+beta*p(k)
    return w


def match_plot(ori, m, n, w, line_name='conjugate_down'):
    title = 'm=' + str(m - 1) + ',train_data=' + str(n)
    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.axis([0, 1, -2, 2])
    # sin(2pix)
    x = np.linspace(0, 1, num=300)
    plt.plot(x, np.sin(2 * np.pi * x), color='y', label='sin(2${\pi}$x)')
    # 拟合曲线
    sol_y = get_func(w, vandermonde(x, m))
    plt.plot(x, sol_y, label=line_name)
    # 散点图
    plt.scatter(ori[0], ori[1], color='r', label='training data')
    plt.legend()
    plt.show()


# 不带正则项
# w = conjugate_down(w0, A, b)
# match_plot((ori_x, ori_y), m, n, w, 'conjugate_down')
# # 带正则项
# la = np.exp(-5)
# A = A + la * np.eye(A.shape[0], A.shape[1])
# w = conjugate_down(w0, A, b)
# match_plot((ori_x, ori_y), m, n, w, 'conjugate_down with correct')

if __name__ == '__main__':
    main()
