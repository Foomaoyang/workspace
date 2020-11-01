# encoding: utf-8
import numpy as np


def is_non_dominated(fit_curr, fit_pop, cursor):
    """
    判断是否为非劣解，当前粒子的适应值fitness_curr与数据集fitness_data进行比较
    :param fit_curr:游标指到要比较的粒子
    :param fit_pop:种群所有的适应度
    :param cursor:
    :return:True：是非支配解
            False：不是非支配解
    """
    for i in range(len(fit_pop)):  # 同种群中的其他粒子挨个比较
        if i == cursor:  # 游标指的粒子是同一粒子，跳过
            continue
        # 如果数据集中存在一个粒子可以完全支配当前解，则证明当前解为劣解，返回False
        if not fit_compare(fit_curr, fit_pop[i]):
            return False  # 返回False说明种群中存在一个粒子可以完全支配游标指到的粒子
    return True


def fit_compare(fitness_curr, fitness_ref):
    """
    判断fitness_curr是否可以被fitness_ref完全支配
    当fitness_curr被fitness_ref完全支配时返回False
    否则说明存在一个维度不会被支配
    :param fitness_curr:
    :param fitness_ref:
    :return:返回True,说明
            返回False,说明种群中存在一个粒子可以完全支配游标指到的粒子
    """
    for i in range(len(fitness_curr)):
        if fitness_curr[i] < fitness_ref[i]:
            return True
    return False


class Pareto:
    def __init__(self, pop, fitness):
        self.pop = pop  # 粒子群坐标信息
        self.fit = fitness  # 粒子群适应值信息
        self.cursor = -1  # 初始化游标位置
        self.pop_size = pop.shape[0]  # 粒子群的数量
        self.bad_num = 0  # 记录劣解的个数

    def next(self):
        # 将游标的位置前移一步，并返回所在检索位的粒子坐标、粒子适应值
        self.cursor = self.cursor + 1
        return self.pop[self.cursor], self.fit[self.cursor]

    def has_next(self):
        # 判断是否已经检查完了所有粒子
        return self.pop_size > self.cursor + 1 + self.bad_num

    def remove(self):
        # 将劣解从数据集删除，避免反复与其进行比较。
        self.fit = np.delete(self.fit, self.cursor, axis=0)
        self.pop = np.delete(self.pop, self.cursor, axis=0)
        # 游标回退一步
        self.cursor = self.cursor - 1
        # 劣解个数，加1
        self.bad_num = self.bad_num + 1

    def pareto_sort(self):
        while self.has_next():
            # 获取当前游标指定位置的粒子信息
            _, fitness_curr = self.next()
            # 判断当前粒子是否pareto最优
            if not is_non_dominated(fitness_curr, self.fit, self.cursor):
                """
                为True：是非支配解，不删除该解
                为False：不是非支配解，删除该粒子
                """
                self.remove()
        return self.pop, self.fit

