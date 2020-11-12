# 梯度法求解多项式曲线拟合
import numpy as np

from lab.funcs import vandermonde, get_Ew, get_RMS, best_answer_without_correct, best_answer_with_correct
from lab.Q6 import generate_data, match_plot


def main():
    m = 3
    n = 15
    test_size = 40
    k = int(1e6)
    # 产生数据
    ori_x, ori_y = generate_data(n)
    test_x, test_y = generate_data(test_size)
    # 确定梯度下降初始化内容，不带正则项
    # w0 = best_answer_without_correct(ori_x, ori_y, m)
    w0 = np.zeros((m, 1))
    x = vandermonde(ori_x, m)
    _x = vandermonde(test_x, m)
    A = x.T @ x
    t = np.mat(ori_y).T
    b = x.T @ t
    # 梯度下降求解w
    w = grad_down(w0, A, b, k, x, t)
    # 画图
    # 计算E_RMS
    e_rms = get_RMS(get_Ew(_x, w, np.mat(test_y).T), test_size)
    print('without correct in test data:e_rms=', e_rms)
    match_plot((ori_x, ori_y), m, n, w, 'grad_down')
    # 梯度下降求解w,带正则项
    la = np.exp(-9)
    A = x.T @ x + la * np.mat(np.eye(A.shape[0], A.shape[1]))
    # w0 = np.mat(best_answer_with_correct(ori_x, ori_y, m, la)).T
    w = grad_down(w0, A, b, k, x, t)
    # 计算E_RMS
    e_rms = get_RMS(get_Ew(_x, w, np.mat(test_y).T), test_size)
    print('with correct in test data:e_rms=', e_rms)
    # 画图
    match_plot((ori_x, ori_y), m, n, w, 'grad_down with correct')


# k 最多的迭代次数
def grad_down(w0, A, b, k, x, t):
    w = np.mat(w0)
    A = np.mat(A)
    b = np.mat(b)
    step_length = 0.001  # 步长
    r = np.mat(b - A * w)  # 梯度反方向
    value1 = get_Ew(x, w, t)  # 第二个
    value0 = value1
    precision = 1e-5
    for _ in range(k):
        w = w + step_length * r
        value0 = value1  # 前一个
        value1 = get_Ew(x, w, t)  # 当前的
        # 如果二次函数取值变大，则减小步长
        if value1 - value0 > 0: step_length = 0.5 * step_length
        # 跳出条件 当loss足够小时
        if value1 < precision:
            print(_)
            break
        r = np.mat(b - A * w)  # 梯度反方向
    return w


def iter_func(A, b, x):
    x = np.mat(x)
    A = np.mat(A)
    b = np.mat(b)
    res = 0.5 * x.T @ A @ x - x.T @ b
    return float(res)


if __name__ == '__main__':
    main()
