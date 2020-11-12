import os

import cv2
import numpy as np

from pca import PCA, psnr


def read_face(file_path):
    file_list = os.listdir(file_path)  # 读取文件夹内文件
    data = []
    show_data = []
    size = (60, 60)  # 压缩原始图片大小以加快处理速度
    for file in file_list:
        path = os.path.join(file_path, file)
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
        img_gray = cv2.resize(img_gray, size)  # 图片压缩
        # 图片混合()
        if len(show_data) == 0:
            show_data = img_gray
        else:
            show_data = np.hstack([show_data, img_gray])
        h, w = img_gray.shape
        data_gray = img_gray.reshape(h * w)  # 图片拉伸
        data.append(data_gray)  # 加入处理数据集中
    cv2.imwrite('original pic.jpg', np.array(show_data))
    cv2.imshow('original picture', np.array(show_data))  # 显示图片
    cv2.waitKey(0)
    return np.mat(np.array(data)).T, size


def pic_handle(img_data, sd, ori_size):
    """
    做PCA处理之后，在还原到原来的维度，然后显示，之后输出信噪比
    """
    Pca = PCA(sd, img_data)
    c_data, w_star = Pca.pca()  # 进行pca降维,获取投影矩阵
    w_star = np.real(w_star)
    print(w_star)
    new_data = w_star * w_star.T * c_data + Pca.mean  # 还原到原来的维度
    total_img = []
    # 图片混合
    for i in range(Pca.data_size):
        if len(total_img) == 0:
            total_img = new_data[:, i].T.reshape(ori_size)
        else:
            total_img = np.hstack([total_img, new_data[:, i].T.reshape(ori_size)])
    # 计算信噪比
    print('信噪比:')
    for i in range(Pca.data_size):
        a = psnr(np.array(data[:, i].T), np.array(new_data[:, i].T))
        print('图', i, '的信噪比为:', a, 'dB')
    # 处理图片
    total_img = np.array(total_img).astype(np.uint8)
    cv2.imwrite('pca image.jpg', total_img)  # 图片显示
    cv2.imshow('pca image', total_img)
    cv2.waitKey(0)


small_d = 2
data, ori_size = read_face('pic')
pic_handle(data, small_d, ori_size)
