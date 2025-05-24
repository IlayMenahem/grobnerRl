import sympy
from grobnerRl.benchmark.benchmark import optimal_vs_standard

if __name__ == '__main__':
    num_episodes = 1000
    step_limit = 250
    ideal_params = [10, 3, 5, 3, sympy.FF(32003), 'lex']
    optimal_vs_standard(num_episodes, *ideal_params)
