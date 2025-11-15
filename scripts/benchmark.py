"""
Benchmark script for comparing MCTS agent with standard selection strategies.
"""

if __name__ == "__main__":
    from grobnerRl.benchmark.benchmark import benchmark_expert
    from grobnerRl.envs.env import BuchbergerEnv
    from grobnerRl.envs.ideals import SAT3IdealGenerator
    from grobnerRl.experts import BasicExpert

    num_vars = 5
    multiplier = 4.7
    num_clauses = int(num_vars * multiplier)

    ideal_dist = SAT3IdealGenerator(num_vars, num_clauses)
    num_episodes = 1000

    env = BuchbergerEnv(ideal_dist)

    buchbergerAgent = BasicExpert(env, strategy="normal")
    benchmark_expert(buchbergerAgent, num_episodes, env, folder="figs")
