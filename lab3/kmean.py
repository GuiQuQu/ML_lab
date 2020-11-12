import numpy as np
import matplotlib.pyplot as plt


def main():
    type_size = 50
    k = 3
    mu = [[-1, 2], [3, 1.5], [0.5, -0.8]]
    train_data = generate_data(k, type_size, mu)
    kmean = KMean(k, train_data, 1e-5, times=100)
    kmean.k_mean()
    print(kmean.mean)
    kmean.get_plot()


def generate_data(k, type_size, mu):
    # 协方差
    one = np.mat(np.array([1, 0]))
    two = np.mat(np.array([0, 1]))
    cov = np.vstack([one, two])
    one_kind = np.mat(np.random.multivariate_normal(mu[0], cov, type_size))
    i = 1
    while i < k:
        two_kind = np.mat(np.random.multivariate_normal(mu[i], cov, type_size))
        one_kind = np.vstack([one_kind, two_kind])
        i = i + 1

    return one_kind.T


class KMean:
    # 初始化
    def __init__(self, k, data, precision, times):
        self.k = k  # 分类超参k类
        self.data_size = data.shape[1]  # 分类数据的个数
        self.n = data.shape[0]  # 特征数据的维度
        self.data = np.vstack([data, np.mat(np.zeros([1, self.data_size]))])  # 分类数据，带标签
        self.mean = None  # 中心点矩阵(n+1)*(k)
        self.precision = precision  # 迭代精度
        self.times = times
        # self.k_mean_res = data

    def k_mean(self):
        # 选择k个点当作起始的mean,get label
        for i in range(self.k):
            self.data[self.n, i] = i  # 标签在self.n这一行上
        self.mean = self.data[:, :self.k]
        old_mean = None
        time = 0
        while True:
            self.get_label()  # 贴标签
            # 重新计算mean
            old_mean = self.mean
            self.mean = self.get_mean()
            print(self.mean)
            print()
            if self.get_mean_diff(old_mean) <= self.precision: break
            time = time + 1
            if time > self.times:
                break

    def get_mean_diff(self, old_mean):
        # 计算中心点之间的差距
        result = 0
        for i in range(self.k):
            result = result + self.mean[:self.n, i].T * old_mean[:self.n, i]
        return result

    def get_label(self):
        labels = [0] * self.data_size
        for i in range(self.data_size):  # 计算每一个点的标签
            distance = float('inf')
            point = self.data[:self.n, i]  # 取第i列，并且不要最后一行标签
            for j in range(self.k):  # 计算该点距离哪个center点最近
                center = self.mean[:self.n, j]  # 取mean中的第j类的中心点
                if self.get_distance(center, point) < 1e-6:
                    distance = 0
                    labels[i] = self.mean[self.n, j]
                elif self.get_distance(center, point) < distance:
                    labels[i] = self.mean[self.n, j]
                    distance = self.get_distance(center, point)
        self.data[self.n, :] = np.mat(labels)

    def get_distance(self, point, center):
        return float((center - point).T * (center - point))

    # 根据已有mean重新计算mean
    def get_mean(self):
        ret = np.mat(np.zeros((self.n, 1)))  # 代表mean,没有加入标签中的一列
        for i in range(self.k):  # 计算第k类的均值(center)
            temp = np.mat(np.zeros((self.n, 1)))  # 第k类的暂时均值，没有加入标签
            count = 0
            for j in range(self.data_size):  # 看第j个点
                if self.data[self.n, j] != self.mean[self.n, i]:  # 标签不同跳过
                    continue
                else:
                    temp = temp + self.data[:self.n, j]
                    count = count + 1
            temp = 1 / count * temp
            if ret.shape == temp.shape and float(ret.T * ret) == 0:
                ret = temp
            else:
                ret = np.hstack([ret, temp])
        # 增加每个中心点的标签
        ret = np.vstack([ret, self.mean[self.n, :]])
        return ret

    def get_class(self):  # 统计每一类分别有多少数据  #-----------------------------
        class_num = [0] * self.k
        for i in range(self.k):
            for j in range(self.data_size):
                if self.data[self.n, j] == self.mean[self.n, i]:
                    class_num[i] = class_num[i] + 1
        return class_num

    def get_plot(self):
        # data
        plt.title('K-mean method')
        color = ['r', 'b', 'g', 'y']
        plt.xlabel('x1')
        plt.ylabel('x2')

        for i in range(self.k):
            data_x1 = []
            data_x2 = []
            for j in range(self.data_size):
                if self.data[self.n, j] == self.mean[self.n, i]:
                    data_x1.append(self.data[0, j])
                    data_x2.append(self.data[1, j])
            plt.scatter(data_x1, data_x2, color=color[i], label='class' + str(i + 1))
        # 中心点
        center_x1 = list(self.mean[0, :])
        center_x2 = list(self.mean[1, :])
        plt.scatter(center_x1, center_x2, marker='+', color='b', label='center')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
