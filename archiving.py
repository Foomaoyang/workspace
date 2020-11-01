import numpy as np
import random

from typing import List


class MeshCrowd(object):
    def __init__(self, pop_size, pop_archiving, fit_archiving, pop_min, pop_max, mesh_div):
        self.pop_size = pop_size
        self.pop_archiving, self.fit_archiving = pop_archiving, fit_archiving
        self.mesh_div = mesh_div  # 等分因子，默认值为10等分
        self.mesh_min, self.mesh_max = pop_min, pop_max
        self.archive_size = self.pop_archiving.shape[0]  # Archive存档中粒子数量
        self.id_archiving = np.zeros(self.archive_size)  # 档案中每个粒子对应在网格里的id编号，检索位与curr_archiving的检索位为相对应
        self.crowd_archiving = np.zeros(self.archive_size)  # 拥挤度矩阵，用于记录当前粒子所在网格的总粒子数，检索位与curr_archiving的检索为相对应
        self.probability_archiving = np.zeros(self.archive_size)  # 各个粒子被选为gbest的概率，检索位与curr_archiving的检索位为相对应
        # TODO 种群的全局最优维度不对  gbest是100行，2列
        self.gbest_pop = np.zeros((self.pop_size, self.pop_archiving.shape[1]))  # 初始化gbest矩阵_坐标和适应值
        self.gbest_fit = np.zeros((self.pop_size, self.fit_archiving.shape[1]))  #

    def divide_archiving(self):
        """
        进行网格划分，为每个粒子定义网格编号
        """
        for i in range(self.archive_size):
            # 得到档案中每个粒子的id编号
            self.id_archiving[i] = self.cal_mesh_id(self.pop_archiving[i])

    def cal_mesh_id(self, individual):
        """
        计算档案中第i个粒子的网格id编号，看Readme.md文档的get_crowd部分
        首先，将每个维度按照等分因子进行等分离散化，获取粒子在各维度上的编号。按照10进制将每一个维度编号等比相加（如过用户自定义了mesh_div_num的值，则按照自定义），计算出值
        :param individual: 档案中第i个粒子
        :return: 档案中第i个粒子对应的网格id编号
        """
        mesh_id = 0
        for i in range(self.pop_archiving.shape[1]):  # 注意是循环次数是由维度确定
            id_dim = int((individual[i] - self.mesh_min[i]) * self.archive_size / (self.mesh_max[i] - self.mesh_min[i]))
            mesh_id = mesh_id + id_dim * (self.mesh_div ** i)
        return mesh_id

    def get_crowd(self):
        """
        计算拥挤度,看Readme.md文档的get_crowd部分
        将档案中的粒子全部编号，从0到archive_size-1。在同一个网格中的粒子，粒子的拥挤度设为粒子的数量。赋值完一个删除一个避免重复
        """
        # 定义一个数组存放粒子集的索引号，用于辅助计算
        # index是[0,1,...,archive_size-1]的列表，
        index = (np.linspace(0, self.archive_size - 1, self.archive_size)).tolist()
        index = map(int, index)
        index = list(index)
        while len(index) > 0:
            index_same = [index[0]]  # 存放本次子循环中与index[0]粒子具有相同网格id所有检索位
            for i in range(1, len(index)):
                if self.id_archiving[index[0]] == self.id_archiving[index[i]]:
                    index_same.append(index[i])
            same_num = len(index_same)  # 档案中其他粒子id号和第一个粒子的id号相同的数量，也即同一网孔中粒子数量
            for i in index_same:  # 计算同一网格中粒子的拥挤度
                self.crowd_archiving[i] = same_num  # 同一网孔里的粒子，它的拥挤度设为网孔中粒子数量。然后存到拥挤度档案中
                index.remove(i)  # 在数组中删除已经赋值过拥挤度的粒子，避免重复计算


class Findgbest(MeshCrowd):
    def __init__(self, pop_size, pop_archiving, fit_archiving, pop_min, pop_max, mesh_div_num):
        super(Findgbest, self).__init__(pop_size, pop_archiving, fit_archiving, pop_min, pop_max, mesh_div_num)
        self.divide_archiving()
        self.get_crowd()

    def get_probability(self):
        for i in range(self.archive_size):
            self.probability_archiving = 10.0 / (self.crowd_archiving ** 3)
        self.probability_archiving = self.probability_archiving / np.sum(self.probability_archiving)  # 所有粒子的被选概率的总和为1

    def get_gbest_index(self):
        random_pro = random.uniform(0.0, 1.0)  # 生成一个0到1之间的随机数
        for i in range(self.archive_size):
            if random_pro <= np.sum(self.probability_archiving[0:i + 1]):
                return i  # 返回检索值

    def get_gbest(self):
        self.get_probability()
        self.gbest_index = self.get_gbest_index()
        for i in range(self.pop_size):
            self.gbest_pop[i] = self.pop_archiving[self.gbest_index]  # gbest矩阵_坐标
            self.gbest_fit[i] = self.fit_archiving[self.gbest_index]  # gbest矩阵_适应值
        return self.gbest_pop, self.gbest_fit


class ClearArchiving(Findgbest):
    def __init__(self, pop_size, pop_archive, fit_archive, min_, max_, mesh_div_num):
        super(Findgbest, self).__init__(pop_size, pop_archive, fit_archive, min_, max_, mesh_div_num)
        self.divide_archiving()
        self.get_crowd()  # 计算网格拥挤度
        self.thresh = None
        self.pop_archive = None
        self.fit_archive = None

    def get_probability(self):
        for i in range(self.archive_size):
            self.probability_archiving = self.crowd_archiving ** 2

    def get_clear_index(self):  # 按概率清除粒子，拥挤度高的粒子被清除的概率越高
        len_clear = self.pop_archive.shape[0] - self.thresh  # 需要清除掉的粒子数量
        clear_index = []
        while len(clear_index) < len_clear:
            #################
            # random_pro = random.uniform(0.0, np.sum(self.probability_archiving))  # 生成一个0到1之间的随机数
            random_pro = random.uniform(0.0, 1.0)  # 生成一个0到1之间的随机数
            for i in range(self.archive_size):
                if random_pro <= np.sum(self.probability_archiving[0:i + 1]):
                    if i not in clear_index:
                        clear_index.append(i)  # 记录检索值
        return clear_index

    def del_pop(self, thresh: int) -> (float, float):
        """
        剔除档案中多余粒子
        当档案中的粒子数量超过thresh阈值时删除一定数量的非支配粒子
        :param thresh: 档案大小
        :return:
        """
        self.thresh = thresh
        self.get_probability()
        del_index = self.get_clear_index()
        self.pop_archive = np.delete(self.pop_archive, del_index, axis=0)
        self.fit_archive = np.delete(self.fit_archive, del_index, axis=0)
        return self.pop_archive, self.fit_archive
