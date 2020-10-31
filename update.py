import numpy as np
import random
import pareto
import archiving


def update_v(pop, pop_p, pop_g, pop_v, v_min, v_max, w, c1, c2):
    """
    更新速度
    """
    v = w * pop_v + c1 * (pop_p - pop) + c2 * (pop_g - pop)
    # 如果粒子的新速度大于最大值，则置为最大值；小于最小值，则置为最小值
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j] < v_min[j]:
                v[i, j] = v_min[j]
            if v[i, j] > v_max[j]:
                v[i, j] = v_max[j]
    return v


def update_pos(pop, pop_v, pop_min, pop_max):
    """
    更新位置
    """
    pop = pop + pop_v
    # 大于最大值，则置为最大值；小于最小值，则置为最小值
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            if pop[i, j] < pop_min[j]:
                pop[i, j] = pop_min[j]
            if pop[i, j] > pop_max[j]:
                pop[i, j] = pop_max[j]
    return pop


def update_pbest(pop, fitness, pop_pbest, fit_pbest):
    """
    更新个体最优
    :return: 返回个体的最优位置和适应度
    """
    for i in range(fit_pbest.shape[0]):
        # 通过比较历史pbest和当前粒子适应值，决定是否需要更新pbest的值。
        if compare_fit(fitness[i], fit_pbest[i]):
            pop_pbest[i] = pop[i]
            fit_pbest[i] = fitness[i]
    return pop_pbest, fit_pbest


def update_gbest(pop_size, pop_archiving, fit_archiving, pop_min, pop_max, mesh_div):
    get_g = archiving.Findgbest(pop_size, pop_archiving, fit_archiving, pop_min, pop_max, mesh_div)
    return get_g.get_gbest()


def update_archive(pop, pop_size, fitness, pop_archive, fit_archive, pop_min, pop_max, thresh, mesh_div):
    """
    种群更新后，经过三次筛选，构建Archive存档；三次筛选在论文中，通过Pareto非支配排序实现
    """
    # 首先，计算当前粒子群的pareto边界，将边界粒子加入到存档archiving中，这些粒子数可能会溢出，需要筛选剔除
    pareto_1 = pareto.Pareto(pop, fitness)
    curr_pop, curr_fit = pareto_1.pareto_sort()

    # 其次，在存档中根据支配关系进行第二轮筛选，将非边界粒子去除
    curr_pop = np.concatenate((pop_archive, curr_pop), axis=0)  # concatenate连接两个序列
    curr_fit = np.concatenate((fit_archive, curr_fit), axis=0)
    pareto_2 = pareto.Pareto(curr_pop, curr_fit)
    curr_pop, curr_fit = pareto_2.pareto_sort()

    # 最后，判断存档数量是否超过了存档阀值。如果超过了阀值，则清除掉一部分（拥挤度高的粒子被清除的概率更大，保证分布性）
    if curr_pop.shape[0] > thresh:
        clear_ = archiving.ClearArchiving(pop_size, curr_pop, curr_fit, pop_min, pop_max, mesh_div)
        curr_pop, curr_fit = clear_.del_pop(thresh)
    return curr_pop, curr_fit


def compare_fit(pop_fit, pbest_fit):
    num_large, num_less = 0, 0
    for i in range(len(pop_fit)):
        if pop_fit[i] > pbest_fit[i]:
            num_large = num_large + 1
        if pop_fit[i] < pbest_fit[i]:
            num_less = num_less + 1
    # 如果当前粒子支配历史pbest，则更新,返回True
    if num_large > 0 and num_less == 0:
        return True
    # 如果历史pbest支配当前粒子，则不更新,返回False
    elif num_large == 0 and num_less > 0:
        return False
    else:
        # 如果互不支配，则按照概率决定是否更新
        random_ = random.uniform(0.0, 1.0)
        if random_ > 0.5:
            return True
        else:
            return False