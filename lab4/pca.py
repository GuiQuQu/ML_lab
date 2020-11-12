import math
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    small_d = 2
    data_size = 50
    mu = [1, 2, 3]
    train_data = generate_data(data_size, mu)
    pca_sol = PCA(small_d, train_data)
    pca_sol.pca()
    pca_sol.get_plot()


def generate_data(size, mu):
    # 协方差
    one = np.mat(np.array([100, 0, 0]))
    two = np.mat(np.array([0, 100, 0]))
    three = np.mat(np.array([0, 0, 1]))
    cov = np.vstack([one, two, three])
    one_kind = np.mat(np.random.multivariate_normal(mu, cov, size))
    return one_kind.T


class PCA(object):
    def __init__(self, small_d, data):
        self.d = data.shape[0]  # 原来的维度
        self.small_d = small_d  # 降维之后的维度
        self.data_size = data.shape[1]
        self.data = data
        self.__pca_data = np.mat(np.zeros((self.small_d, self.data_size)))
        self.mean = np.mean(self.data, 1)

    def pca(self):
        c_data = self.__centralization()  # 中心化
        cov = c_data * c_data.T  # 数据的协方差矩阵(由于去中心化使得协方差不需要减去均值)
        eig_values, eig_vectors = np.linalg.eig(cov)  # 特征值分解
        eig_index = np.argsort(eig_values)  # 排序
        w_star = eig_vectors[:, eig_index[:-(self.small_d + 1):-1]]  # 寻找top-small_d大的维度，作为主成分
        self.__pca_data = w_star * w_star.T * c_data + self.mean  # 利用特征矩阵对数据降维
        # self.__pca_data = w_star * w_star.T * c_data  # 利用特征矩阵对数据降维
        return c_data, w_star  # 返回特征矩阵

    def get_plot(self):
        # 画数据
        xs = []
        ys = []
        zs = []
        small_xs = []
        small_ys = []
        small_zs = []
        for i in range(self.data_size):
            xs.append(float(self.data[0, i]))
            ys.append(float(self.data[1, i]))
            zs.append(float(self.data[2, i]))
            small_xs.append(float(self.__pca_data[0, i]))
            small_ys.append(float(self.__pca_data[1, i]))
            small_zs.append(float(self.__pca_data[2, i]))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(xs, ys, zs, label='sample data', color='b')
        ax.scatter(small_xs, small_ys, small_zs, label='PCA data', color='r')
        ax.legend()
        ax.view_init(90, 0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def __centralization(self):  # 中心化
        mean = np.mean(self.data, 1)
        return self.data - mean


# 计算信噪比(越大越好)
def psnr(img1, img2):  # img1 原始图片 img2 处理图片(信息部分)
    img1 = np.real(img1)
    img2 = np.real(img2)
    noise = img1 - img2  # 噪声部分
    # 计算img2 和noise的方差
    var_info = np.mean((img2 - np.mean(img2)) ** 2)
    var_noise = np.mean((noise - np.mean(noise)) ** 2)
    snr = var_info / var_noise
    snr_dB = math.log10(snr)
    return snr_dB


if __name__ == '__main__':
    main()
