"""
Benchmark script for comparing MCTS agent with standard selection strategies.
"""

from grobnerRl.envs.deepgroebner import BuchbergerEnv, MCTSAgent, BuchbergerAgent, OracleAgent
from grobnerRl.benchmark.benchmark import benchmark_agent

if __name__ == '__main__':
    ideal_dist = '3-4-4-uniform'
    num_episodes = 250

    env = BuchbergerEnv(ideal_dist=ideal_dist, mode='eval')

    orcaleAgent = OracleAgent(env)
    benchmark_agent(orcaleAgent, num_episodes, env, folder='figs', agent_name='OracleAgent')

    buchbergerAgent = BuchbergerAgent('normal')
    benchmark_agent(buchbergerAgent, num_episodes, env, folder='figs', agent_name='BuchbergerAgent')

    MCTSagent = MCTSAgent(env, n_simulations=50, c=1, gamma=0.99, rollout_policy='normal')
    benchmark_agent(MCTSagent, num_episodes, env, folder='figs', agent_name='MCTSAgent')


    env.close()
