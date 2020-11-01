# encoding: utf-8
import numpy as np


# 为了便于图示观察，试验测试函数为二维输入、二维输出
# 适应值函数：实际使用时请根据具体应用背景自定义
def test_func(individual):
    """
    适应度函数
    :param individual: 单个粒子，不是整个种群
    :return: 粒子的适应度值
    """
    degree_45 = ((individual[0] - individual[1]) ** 2 / 2) ** 0.5
    degree_135 = ((individual[0] + individual[1]) ** 2 / 2) ** 0.5
    fit_1 = 1 - np.exp(-degree_45 ** 2 / 0.5) * np.exp(-(degree_135 - np.sqrt(200)) ** 2 / 250)
    fit_2 = 1 - np.exp(-degree_45 ** 2 / 5) * np.exp(-degree_135 ** 2 / 350)
    return [fit_1, fit_2]
