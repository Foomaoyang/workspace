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
        self.max_v = (pop_min - pop_max) * 0.05  # 速度下限
        self.min_v = (pop_min - pop_max) * 0.05 * (-1)  # 速度上限
        self.plot_ = plot.POF()
        self.pop, self.pop_v = None, None
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
        self.pop_archive, self.fit_archive = init.init_archive(self.pop, self.fitness)
        # 初始化全局最优
        self.pop_g, self.fit_g = init.init_gbest(self.pop_archive, self.fit_archive, self.mesh_div, self.pop_min,
                                                 self.pop_max, self.pop_size)

    def evaluation_fitness(self):
        """
        计算适应度值
        :return:
        """
        fitness_curr = []
        for i in range(self.pop.shape[0]):
            fitness_curr.append(test_func(self.pop[i]))
        self.fitness = np.array(fitness_curr)  # 适应值

    def pop_update(self):
        """
        更新粒子位置、速度、适应值、个体最优、全局最优、外部存档
        :return:
        """
        self.pop_v = update.update_v(self.pop, self.pop_p, self.pop_g, self.pop_v, self.min_v, self.max_v,
                                     self.w, self.c1, self.c2)
        self.pop = update.update_pos(self.pop, self.pop_v, self.pop_min, self.pop_max)
        self.evaluation_fitness()
        self.pop_p, self.fit_p = update.update_pbest(self.pop, self.fitness, self.pop_p, self.fit_p)
        self.pop_archive, self.fit_archive = update.update_archive(self.pop, self.pop_size, self.fitness,
                                                                   self.pop_archive, self.fit_archive, self.pop_min,
                                                                   self.pop_max, self.thresh, self.mesh_div)
        self.pop_g, self.fit_g = update.update_gbest(self.pop_size, self.pop_archive, self.fit_archive, self.pop_min,
                                                     self.pop_max, self.mesh_div)

    def run(self, iterations):
        self.initialize()
        self.plot_.draw(self.pop, self.fitness, self.pop_archive, self.fit_archive, -1)
        for i in range(iterations):
            self.pop_update()  # 种群更新
            self.plot_.draw(self.pop, self.fitness, self.pop_archive, self.fit_archive, i)
        return self.pop_archive, self.fit_archive
