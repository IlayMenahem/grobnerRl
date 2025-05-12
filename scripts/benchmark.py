from sympy.polys.domains import ZZ
from grobnerRl.benchmark.benchmark import benchmark_agent

if __name__ == '__main__':
    num_episodes = 1000
    step_limit = 500
    ideal_params = [5, 7, 15, 3, ZZ, 'grevlex']

    benchmark_agent('degree_after_reduce', num_episodes, step_limit, *ideal_params)
