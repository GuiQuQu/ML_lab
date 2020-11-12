# 解析法求解带正则项的多项式曲线拟合
import numpy as np
import matplotlib.pyplot as plt
from lab.funcs import best_answer_with_correct, generate_data, vandermonde, get_Ew, match_plot, get_RMS

m = 9  # 阶数 阶数从0 到 m-1
n = 15  # 训练数据个数
test_size = 45  # 测试数据个数
la = np.exp(-5)  # lambda
# 生成数据

x0 = np.linspace(0, 2 * np.pi, num=n, endpoint=True)
y0 = np.sin(2 * x0)
noise = np.random.randn(n) * 0.1
y = y0 + noise
# ori = generate_data(n)
ori = (x0, y)
test_data = generate_data(test_size)
# 求解
w = best_answer_with_correct(ori[0], ori[1], m, l_lambda=la)
match_plot(ori, m, n, w, "solution with correct")

# 确定最好的lambda
# 通过画图来寻找近似的E(w)最低的点  绘制 train_RMS-lambda  和 test_RMS-lambda 图像


la_list = []
train_RMS = []
test_RMS = []
for _ in np.arange(-15, 0):
    la_list.append(np.exp(_))
for lal in la_list:
    _w = np.array(best_answer_with_correct(ori[0], ori[1], m, lal))
    _w = np.reshape(_w, (m, 1))
    _x = vandermonde(ori[0], m)
    _t = np.array(ori[1]).reshape(ori[1].shape[0], 1)
    train_RMS.append(get_RMS(float(get_Ew(_x, _w, _t)), n))
    _x = vandermonde(test_data[0], m)
    _t = np.array(test_data[1]).reshape(test_data[1].shape[0], 1)
    test_RMS.append(get_RMS(float(get_Ew(_x, _w, _t)), test_size))

# RMS-lambda图像
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('$E_{RMS}$')
plt.xticks([np.exp(-i) for i in range(5, 15, 2)], ['$e^{' + str(-i) + '}$' for i in range(5, 15, 2)])
plt.plot(la_list, train_RMS, color='r', label='train')
plt.plot(la_list, test_RMS, color='b', label='test')
plt.legend()
plt.show()
