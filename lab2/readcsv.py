# 读取scv文件
import csv
import numpy as np


def main():
    read_csv('divorce.csv')


def read_csv(file_name):
    f = csv.reader(open(file_name, 'r'))
    x_list = []
    y_list = []
    row_num = 0
    for i in f:
        # 将其分隔开
        row_list = i[0].split(';')
        if row_num > 0:
            # 转化成数字
            num_list = []
            for value in row_list:
                num_list.append(int(value))
            # 提取x和y
            one_row = np.mat(num_list)
            column = one_row.shape[1]
            x_list.append(one_row[0, 0:column - 2].T)

            y_list.append(int(one_row[0, column - 1]))
        row_num = row_num + 1
    # print(x_list[0].shape)
    # print(len(x_list))
    # print(len(y_list))
    return x_list, y_list


if __name__ == '__main__':
    main()
