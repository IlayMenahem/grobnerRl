from grobnerRl.data import generate_data
from grobnerRl.envs.ideals import parse_ideal_dist

if __name__ == "__main__":
    num_vars = 3
    max_degree = 4
    num_polynomials = 4
    step_limit = 10000
    size = 1000
    path = 'data/optimal_reductions.json'
    generator = parse_ideal_dist(f'{num_vars}-{max_degree}-{num_polynomials}-uniform')

    generate_data(generator, step_limit, size, path)
