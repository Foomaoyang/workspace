# encoding: utf-8
import numpy as np


def is_non_dominated(fitness_curr, fitness_data, cursor):
    """
    判断是否为非劣解，当前粒子的适应值fitness_curr与数据集fitness_data进行比较
    :param fitness_curr:
    :param fitness_data:
    :param cursor:
    :return:
    """
    for i in range(len(fitness_data)):
        if i == cursor:  # 游标指的粒子是同一粒子，跳过
            continue
        # 如果数据集中存在一个粒子可以完全支配当前解，则证明当前解为劣解，返回False
        if not fit_compare(fitness_curr, fitness_data[i]):
            return False
    return True


def fit_compare(fitness_curr, fitness_ref):
    """
    判断fitness_curr是否可以被fitness_ref完全支配
    当fitness_curr被fitness_ref完全支配时返回False
    否则说明存在一个维度不会被支配
    :param fitness_curr:
    :param fitness_ref:
    :return:
    """
    for i in range(len(fitness_curr)):
        if fitness_curr[i] < fitness_ref[i]:
            return True
    return False


class Pareto:
    def __init__(self, pop, fitness):
        self.pop_data = pop  # 粒子群坐标信息
        self.fitness_data = fitness  # 粒子群适应值信息
        self.cursor = -1  # 初始化游标位置
        self.len_ = pop.shape[0]  # 粒子群的数量
        self.bad_num = 0  # 劣解的个数

    def next(self):
        # 将游标的位置前移一步，并返回所在检索位的粒子坐标、粒子适应值
        self.cursor = self.cursor + 1
        return self.pop_data[self.cursor], self.fitness_data[self.cursor]

    def has_next(self):
        # 判断是否已经检查完了所有粒子
        return self.len_ > self.cursor + 1 + self.bad_num

    def remove(self):
        # 将劣解从数据集删除，避免反复与其进行比较。
        self.fitness_data = np.delete(self.fitness_data, self.cursor, axis=0)
        self.pop_data = np.delete(self.pop_data, self.cursor, axis=0)
        # 游标回退一步
        self.cursor = self.cursor - 1
        # 劣解个数，加1
        self.bad_num = self.bad_num + 1

    def pareto_sort(self):
        while self.has_next():
            # 获取当前位置的粒子信息
            _, fitness_curr = self.next()
            # 判断当前粒子是否pareto最优
            if not is_non_dominated(fitness_curr, self.fitness_data, self.cursor):
                self.remove()
        return self.pop_data, self.fitness_data

