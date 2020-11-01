# encoding: utf-8
import numpy as np
from fitness_funs import *
import init
import update
import plot


class Mopso:
    def __init__(self, pop_size, w, c1, c2, pop_min, pop_max, thresh, mesh_div=10):
        """

        :param pop_size: 粒子群种群数量
        :param w: 惯性系数
        :param c1: 社会影响因子
        :param c2: 个体影响因子
        :param pop_min: 搜索范围
        :param pop_max: 搜索范围
        :param thresh: 外部存档Archive数量
        :param mesh_div: 网格法等分数量
        """
        self.w, self.c1, self.c2 = w, c1, c2
        self.pop_size = pop_size
        self.thresh, self.mesh_div = thresh, mesh_div
        self.pop_min, self.pop_max = pop_min, pop_max
        self.max_v = (pop_max - pop_min) * 0.05  # 速度上限
        self.min_v = self.max_v * (-1)  # 速度下限
        self.POF = plot.POF()
        self.pop, self.pop_v = None, None  # 种群粒子的位置和速度
        self.pop_p, self.pop_g = None, None
        self.fit_p, self.fit_g = None, None
        self.fitness = None
        self.pop_archive, self.fit_archive = None, None

    def initialize(self):
        # 初始化粒子坐标和速度
        self.pop, self.pop_v = init.init_pop(self.pop_size, self.pop_min, self.pop_max, self.min_v, self.max_v)
        # 计算适应值
        self.evaluation_fitness()
        # 初始化个体最优
        self.pop_p, self.fit_p = init.init_pbest(self.pop, self.fitness)
        # 初始化外部存档
        # TODO 初始化时外部存档数量太少
        self.pop_archive, self.fit_archive = init.init_archive(self.pop, self.fitness)
        # 初始化全局最优
        # TODO 初始化时全局最优维度不对
        self.pop_g, self.fit_g = init.init_gbest(self.pop_archive, self.fit_archive, self.mesh_div, self.pop_min,
                                                 self.pop_max, self.pop_size)

    def evaluation_fitness(self):
        """
        计算适应度值
        :return:
        """
        fit_temp = []
        for i in range(self.pop.shape[0]):
            fit_temp.append(test_func(self.pop[i]))  # 计算粒子i的适应度
        self.fitness = np.array(fit_temp)  # 适应值

    def pop_update(self):
        """
        更新粒子位置、速度、适应值、个体最优、全局最优、外部存档
        核心程序
        :return:
        """
        # 更新种群粒子速度
        self.pop_v = update.update_v(self.pop, self.pop_p, self.pop_g, self.pop_v, self.min_v, self.max_v,
                                     self.w, self.c1, self.c2)
        # 更新种群粒子位置，顺序不能颠倒
        self.pop = update.update_pos(self.pop, self.pop_v, self.pop_min, self.pop_max)
        # 计算种群适应度值
        self.evaluation_fitness()
        # 更新种群中每个粒子的个体最优
        self.pop_p, self.fit_p = update.update_pbest(self.pop, self.fitness, self.pop_p, self.fit_p)
        # 更新档案
        self.pop_archive, self.fit_archive = update.update_archive(self.pop, self.fitness, self.pop_size,
                                                                   self.pop_archive, self.fit_archive, self.pop_min,
                                                                   self.pop_max, self.thresh, self.mesh_div)
        # 更新全局最优
        # TODO 应该是一行两列,gbest维度不对,改为一行两列
        self.pop_g, self.fit_g = update.update_gbest(self.pop_size, self.pop_archive, self.fit_archive, self.pop_min,
                                                     self.pop_max, self.mesh_div)

    def run(self, iterations):
        """
        调用run()方法进行种群迭代
        :param iterations:
        :return:返回迭代后的档案，在主程序中打印档案中的粒子，也即Pareto曲线的值
        """
        self.initialize()
        self.POF.draw(self.pop, self.fitness, self.pop_archive, self.fit_archive, -1)
        for i in range(iterations):
            self.pop_update()  # 种群更新
            self.POF.draw(self.pop, self.fitness, self.pop_archive, self.fit_archive, i)
        return self.pop_archive, self.fit_archive
