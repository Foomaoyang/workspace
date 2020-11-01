# encoding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fitness_funs


class POF:
    """
    绘制pareto曲线的类
    """
    def __init__(self):
        # 绘制测试函数的曲面，（x1，x2）表示两位度的输入，（y1，y2）表示两位的适应值，
        self.x1 = np.linspace(-5, 5, 50)  # 0到10之间100个点
        self.x2 = np.linspace(-5, 5, 50)
        self.x1, self.x2 = np.meshgrid(self.x1, self.x2)
        self.m, self.n = np.shape(self.x1)
        self.y1, self.y2 = np.zeros((self.m, self.n)), np.zeros((self.m, self.n))
        # 将测试函数绘制出来
        for i in range(self.m):
            for j in range(self.n):
                [self.y1[i, j], self.y2[i, j]] = fitness_funs.test_func([self.x1[i, j], self.x2[i, j]])
        if not os.path.exists('./img_txt'):
            os.makedirs('./img_txt')
            print('创建文件夹img_txt:保存粒子群每一次迭代的图片')

    def draw(self, pop, fit, pop_archive, fit_archive, i):
        """
        绘图方法
        共3个子图，第1、2子图绘制输入坐标与适应值关系，第3图展示pareto边界的形成过程
        :param pop:
        :param fit:
        :param pop_archive:
        :param fit_archive:
        :param i: 迭代次数，第几代
        :return:
        """
        fig = plt.figure(13, figsize=(17, 5))  # 图片排列为一行三列

        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_xlabel('input_x1')
        ax1.set_ylabel('input_x2')
        ax1.set_zlabel('fit_y1')
        ax1.plot_surface(self.x1, self.x2, self.y1, alpha=0.6)  # 测试函数曲面图
        ax1.scatter(pop[:, 0], pop[:, 1], fit[:, 0], s=20, c='blue', marker=".")
        ax1.scatter(pop_archive[:, 0], pop_archive[:, 1], fit_archive[:, 0], s=50, c='red', marker=".")

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_xlabel('input_x1')
        ax2.set_ylabel('input_x2')
        ax2.set_zlabel('fit_y2')
        ax2.plot_surface(self.x1, self.x2, self.y2, alpha=0.6)  # 测试函数曲面图
        ax2.scatter(pop[:, 0], pop[:, 1], fit[:, 1], s=20, c='blue', marker=".")
        ax2.scatter(pop_archive[:, 0], pop_archive[:, 1], fit_archive[:, 1], s=50, c='red', marker=".")

        ax3 = fig.add_subplot(133)
        ax3.set_xlim((0, 25))
        ax3.set_ylim((0, 25))
        ax3.set_xlabel('fitness_y1')
        ax3.set_ylabel('fitness_y2')
        ax3.scatter(fit[:, 0], fit[:, 1], s=10, c='blue', marker=".")
        ax3.scatter(fit_archive[:, 0], fit_archive[:, 1], s=30, c='red', marker=".", alpha=1.0)
        # plt.show()
        plt.savefig('./img_txt/第' + str(i + 1) + '次迭代.png')
        print('第' + str(i + 1) + '次迭代的图片保存于 img_txt 文件夹')
        plt.close()
