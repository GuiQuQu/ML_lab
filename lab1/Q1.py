# 解析法求解无正则项多项式拟合曲线
# from main import
import matplotlib.pyplot as plt
import numpy as np
from lab.funcs import best_answer_without_correct, generate_data, get_func, vandermonde, get_Ew, get_RMS, match_plot

m = 12  # 阶数从0 到 m-1
n = 15  # 训练数据个数
test_size = 45
# 生成训练数据
ori = generate_data(n)

# 求解
w = best_answer_without_correct(ori[0], ori[1], m)

# 拟合曲线
match_plot(ori, m, n, w, 'solution')


# 绘制loss-m图像,解释过拟合现象  m-w-E(w)
title = 'train_size=' + str(n) + ',test_size=' + str(test_size)
plt.title(title)
plt.xlabel('m')
plt.ylabel('$E_{RMS}$')
plt.xlim(0, 11)
m_list = np.arange(0, 12)
RMS_test = []
RMS_train = []
test_data = generate_data(test_size)
test = np.array(test_data[1]).reshape(test_data[1].shape[0], 1)
train = np.array(ori[1]).reshape(ori[1].shape[0], 1)
for _m in m_list:
    _w = best_answer_without_correct(ori[0], ori[1], _m + 1)
    _w = np.reshape(_w, (_m + 1, 1))
    _x = x = vandermonde(test_data[0], _m + 1)
    RMS_test.append(get_RMS(float(get_Ew(_x, _w, test)), test_size))
    _x = x = vandermonde(ori[0], _m + 1)
    RMS_train.append(get_RMS(float(get_Ew(_x, _w, train)), n))
plt.plot(m_list, RMS_test, color='b', label='test set', marker='o')
plt.plot(m_list, RMS_train, color='r', label='train set', marker='o')
plt.legend()
plt.show()
