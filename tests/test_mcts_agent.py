"""
Test script for the MCTSAgent implementation.
"""

from grobnerRl.envs.deepgroebner import BuchbergerEnv, MCTSAgent


def test_mcts_agent():
    """Test that MCTSAgent works with the Gymnasium interface."""
    # Create environment
    env = BuchbergerEnv(ideal_dist='3-4-4-uniform', mode='eval')
    
    # Create MCTS agent with fewer simulations for faster testing
    agent = MCTSAgent(env, n_simulations=10, c=1.0, gamma=0.99)
    
    # Reset environment
    state, info = env.reset(seed=42)
    G, P = state

    terminated = False
    truncated = False   
 
    while not (terminated or truncated):
        action = agent.act(state)
        
        if action is None:
            break
        
        assert action in P, f"Selected action {action} should be in pair set {P}"
        state, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    test_mcts_agent()
