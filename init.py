# encoding: utf-8
import random
import numpy as np
import archiving
import pareto


def init_pop(pop_size, pop_min, pop_max, v_min, v_max):
    """
    根据可行解的范围加随机数构造初始种群pop解的位置和速度
    :param pop_size: 种群数量大小
    :param pop_min: 可行解搜索范围
    :param pop_max: 可行解搜索范围
    :param v_min:
    :param v_max:
    :return: 返回种群pop初始位置
    """
    pop_dim = len(pop_max)  # 输入参数的维度
    pop = np.zeros((pop_size, pop_dim))
    pop_v = np.zeros((pop_size, pop_dim))
    for i in range(pop_size):
        for j in range(pop_dim):
            pop[i, j] = random.uniform(0, 1) * (pop_max[j] - pop_min[j]) + pop_min[j]
            pop_v[i, j] = random.uniform(0, 1) * (v_max[j] - v_min[j]) + v_min[j]
    return pop, pop_v


def init_pbest(pop, fitness):
    """
    初始化时选择初始位置为个体最优
    :param pop: 粒子位置
    :param fitness: 粒子适应度值
    :return: 返回元组
    """
    return pop, fitness


def init_gbest(pop_archive, fit_archive, mesh_div, pop_min, pop_max, pop_size):
    get_g = archiving.Findgbest(pop_size, pop_archive, fit_archive, pop_min, pop_max, mesh_div)
    # TODO 初始化时全局最优粒子个数不对
    return get_g.get_gbest()


def init_archive(pop, fit):
    # 初始化时存档数量太少
    return pareto.Pareto(pop, fit).pareto_sort()

