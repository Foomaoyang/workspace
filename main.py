# encoding: utf-8
import numpy as np
from mopso import *


def main():
    w = 0.4  # 原来是0.8
    c1 = c2 = 0.1
    pop_size = 50  # 原来100
    iterations = 10  # 30
    mesh_div, thresh = 20, 100  # 原来10, 300
    pop_min, pop_max = np.array([-5, -5]), np.array([5, 5])
    pso = Mopso(pop_size, w, c1, c2, pop_min, pop_max, thresh, mesh_div)  # 粒子群实例化
    pareto_pop, pareto_fitness = pso.run(iterations)  # 经过iterations轮迭代后，pareto边界粒子
    np.savetxt("./img_txt/pareto_in.txt", pareto_pop)  # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/pareto_fitness.txt", pareto_fitness)  # 打印pareto边界粒子的适应值
    print("\n", "pareto边界的坐标保存于：/img_txt/pareto_in.txt")
    print("pareto边界的适应值保存于：/img_txt/pareto_fitness.txt")
    print("\n", "迭代结束,over")


if __name__ == "__main__":
    main()

