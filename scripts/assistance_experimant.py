from grobnerRl.benchmark.benchmark import benchmark_agent, benchmark_assistanted_game
from grobnerRl.envs.ideals import parse_ideal_dist

if __name__ == "__main__":
    num_vars = 3
    max_degree = 20
    num_polynomials = 10
    step_limit = 10000
    size = 1000
    generator = parse_ideal_dist(f'{num_vars}-{max_degree}-{num_polynomials}-uniform')

    benchmark_agent('normal', size, step_limit, benchmark_assistanted_game, generator)
