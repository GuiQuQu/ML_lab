import numpy as np
import matplotlib.pyplot as plt

from kmean import generate_data


def main():
    type_size = 50
    k = 3
    mu = [[-1, 2], [3, 1.5], [0.5, -0.8]]
    train_data = generate_data(k, type_size, mu)
    mix_gau = MixtureGauss(k, train_data, 1e-5, 50)
    mix_gau.parse_param()
    mix_gau.get_plot()
    print(mix_gau.mu)


class MixtureGauss:
    def __init__(self, k, data, precision, times):
        self.k = k
        self.data_size = data.shape[1]  # 分类数据的个数
        self.d = data.shape[0]  # 特征数据的维度
        self.data = data
        self.label = [0] * self.data_size  # 每个数据的标签
        self.precision = precision
        self.times = times
        # 初值
        self.mu = [np.mat(np.zeros((self.d, 1)))] * self.k  # 有k个mu，每个mu都是d维的
        for _k in range(self.k):
            self.mu[_k] = self.mu[_k] + self.data[:, _k]
        self.pi = [1 / k] * self.k  # 有k个pi,每个pi都是数字
        self.sigma = [np.mat(np.eye(self.d))] * self.k  # 有k个sigma,每个sigma都是dxd的矩阵
        self.gamma = np.mat(np.zeros((self.k, self.data_size)))  # 包括所有r(znk)的矩阵

    def parse_param(self):
        # 参数在初始化在init中完成
        self.e_step()
        old_value = 0
        new_value = self.log_likelihood()
        time = 0
        while True:
            for i in range(self.k):  # 计算每一类的参数
                self.m_step(i)
            self.e_step()
            time = time + 1
            print(time)
            print(self.gamma)
            print()
            old_value = new_value
            new_value = self.log_likelihood()
            if np.abs(new_value - old_value) < self.precision: break
            if time > self.times: break
        self.get_label()

    def e_step(self):  # 计算gamma的每一项
        for i in range(self.data_size):  # 对每一个样本
            for j in range(self.k):  # 对每一个类别
                self.gamma[j, i] = self.get_posterior(j, i)

    def m_step(self, k):  # 根据e_step的计算结果更新参数,更新了第k类的参数
        Nk = self.get_Nk(k)
        # 计算mu[k]
        mu_temp = np.mat(np.zeros((self.mu[k].shape[0], self.mu[k].shape[1])))
        for data_num in range(self.data_size):
            mu_temp = mu_temp + self.gamma[k, data_num] * self.data[:, data_num]
        mu_temp = 1 / Nk * mu_temp
        # 计算 sigma[_k]
        sigma_temp = np.mat(np.zeros((self.sigma[k].shape[0], self.sigma[k].shape[1])))
        for data_num in range(self.data_size):
            sigma_temp = sigma_temp + self.gamma[k, data_num] * (self.data[:, data_num] - mu_temp) * (
                    self.data[:, data_num] - mu_temp).T
        sigma_temp = 1 / Nk * sigma_temp
        # 计算pi[_k]
        pi_temp = Nk / self.data_size
        # 更新
        self.mu[k] = mu_temp
        self.sigma[k] = sigma_temp
        self.pi[k] = pi_temp

    def get_gauss(self, mu, sigma, x):
        part1 = 1 / np.power(2 * np.pi, self.d / 2)
        part2 = 1 / np.power(np.linalg.det(sigma), 0.5)
        part3 = np.exp(-0.5 * (x - mu).T * sigma * np.linalg.inv(sigma) * (x - mu))
        return part1 * part2 * part3

    def get_posterior(self, k, data_num):  # 得到某个样本对于分类k的后验
        numerator = self.pi[k] * self.get_gauss(self.mu[k], self.sigma[k], self.data[:, data_num])
        denominator = self.get_px(data_num)
        return numerator / denominator

    def get_Nk(self, k):  # 所有样本对于分类k的后验的累加和
        sum = 0
        for i in range(self.data_size):  # 第k行所有项求和
            sum = sum + self.gamma[k, i]
        return sum

    def get_px(self, data_num):  # 贝叶斯公式的分母，同时也是x_n的概率计算公式
        result = 0
        for _k in range(self.k):
            result = result + self.pi[_k] * self.get_gauss(self.mu[_k], self.sigma[_k], self.data[:, data_num])
        return result

    def log_likelihood(self):
        result = 0
        for data_num in range(self.data_size):
            sum = 0
            for k in range(self.k):
                sum = sum + self.pi[k] * self.get_gauss(self.mu[k], self.sigma[k], self.data[:, data_num])
            result = result + np.log(sum)
        return result

    def get_label(self):
        for data_num in range(self.data_size):  # 计算每一个元素的分类
            max = 0
            for _k in range(self.k):
                if self.gamma[_k, data_num] > self.gamma[max, data_num]:
                    max = _k
            self.label[data_num] = max

    def get_class_num(self):
        class_num = [0] * self.k
        for i in range(self.k):
            for j in range(self.data_size):
                if self.label[j] == i:
                    class_num[i] = class_num[i] + 1
        return class_num

    def get_plot(self):
        title = 'data_size:' + str(self.data_size) + ',iter num:' + str(self.times)
        plt.title(title)
        color = ['r', 'b', 'g', 'y']
        plt.xlabel('x1')
        plt.ylabel('x2')
        for i in range(self.k):
            data_x1 = []
            data_x2 = []
            for j in range(self.data_size):
                if self.label[j] == i:
                    data_x1.append(self.data[0, j])
                    data_x2.append(self.data[1, j])
            plt.scatter(data_x1, data_x2, color=color[i], label='class' + str(i + 1))
        # 中心点
        center_x1 = []
        center_x2 = []
        for k in range(self.k):
            center_x1.append(float(self.mu[k][0, 0]))
            center_x2.append(float(self.mu[k][1, 0]))
        plt.scatter(center_x1, center_x2, marker='+', color='b', label='center')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
