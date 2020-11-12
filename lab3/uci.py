# http://archive.ics.uci.edu/ml/datasets/DrivFace
# 只通过front类别的数据将这些front类别的数据分为来自4个司机的4类

import numpy as np

from mixtureGauss import MixtureGauss
from kmean import KMean


def main():
    k = 4
    train_data = get_data(k)
    # k_mean(k, train_data, 1e-5, 100)
    gmm(k, train_data, 1e-5, 30)


def get_rate(ori_label, rate_label):  # 准确率评估
    a = 0
    b = 0
    c = 0
    d = 0
    data_size = ori_label.shape[1]
    for i in range(data_size):
        for j in range(data_size):
            if i < j:
                if rate_label[0, i] == rate_label[0, j]:
                    if ori_label[0, i] == ori_label[0, j]:
                        a = a + 1
                    else:
                        b = b + 1
                else:
                    if ori_label[0, i] == ori_label[0, j]:
                        c = c + 1
                    else:
                        d = d + 1
    return 2 * (a + d) / (data_size * (data_size - 1))

def k_mean(k, train_data, precision, times):
    ori_label = train_data[0, :]
    kmean = KMean(k, train_data[1:, :], precision, times)
    kmean.k_mean()
    rate_label = kmean.data[kmean.n, :]
    rate = get_rate(ori_label, rate_label)
    print('kmean rank Index:', rate)

def gmm(k, train_data, precision, times):
    ori_label = train_data[0, :]
    data = data_normal(train_data[1:, :])
    gmm = MixtureGauss(k, data, precision, times)
    gmm.parse_param()
    rate_label = np.mat(gmm.label)
    rate = get_rate(ori_label, rate_label)
    print('GMM rank Index=', rate)


def data_normal(data):
    row = data.shape[0]
    column = data.shape[1]
    ret = []
    for i in range(row):
        max = data[i, 0]
        min = data[i, 0]
        for j in range(column):
            if max < data[i, j]:
                max = data[i, j]
            if min > data[i, j]:
                min = data[i, j]
        for j in range(column):
            a = 5 * (data[i, j] - min) / (max - min)
            # data[i, j] = (data[i, j] - min) / (max - min)
            ret.append(a)
    ret = np.mat(np.array(ret).reshape((row, column)))
    return ret


def get_data(k):  # 包含了标签的数据
    with open('drivPoints.txt', 'r') as file:
        file.readline()
        first_line = file.readline().split(',')
        ret = []
        for i in range(len(first_line)):
            if i == 0 or i == 2:  # filename and imgNum
                continue
            if i == 3:  # label
                continue
            if i == 1:
                ret.append(int(first_line[i]) - 1)
            else:
                ret.append(int(first_line[i]))
        ret = np.mat(np.array(ret))
        while True:
            line_list = file.readline().split(',')
            if len(line_list) <= 1:
                break
            if line_list[3] != '2':
                continue
            x = []
            for i in range(len(line_list)):
                if i == 0 or i == 2:  # filename and imgNum
                    continue
                if i == 3:  # label
                    continue
                if i == 1:
                    x.append(int(line_list[i]) - 1)
                else:
                    x.append(int(line_list[i]))
            x = np.mat(np.array(x))
            ret = np.vstack([ret, x])
        return ret.T


if __name__ == '__main__':
    main()
