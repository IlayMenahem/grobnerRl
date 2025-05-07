# presentation
talking points
1. the definition of MDP, and reinforcement learning
2. the interactive example of coin toss, and it's optimal policy
3. basic definitions in RL
4. algorithm by paradigm DQN, BC, policy gradient, and actor critic

## background

### Markov Decision Processes (MDPs)
Prototypical example
blackjack, or coin toss

definition
the Markov decision processes (MDPs) is a tuple $(\rho,S, T, A, P, R, \gamma)$
S - set of states
T - set of terminal states
$\rho$ - initial state distribution
A(s) - set of actions for state s
$P(s_{t+1}|s_t,a_t)$ - transition probability distribution
$R(s_t,a_t,s_{t+1})$ - reward function
$\gamma$ - discount factor

the objective
find the optimal policy $\pi^*(a|s)$ that maximizes $E[\sum_{k=0}^\infty \gamma^k R_{a_t}(s_t,s_{t+1})]$ where $\gamma \in [0,1)$ is the discount factor.

important notes
- $\gamma$ is usually used to improve the learning process ,rather than being a neutral part of the model. the improvement is due to the fact that it allows the agent to place greater emphasis on immediate rewards which there is more certainty about.
- the set of allowable actions can change by state

### planning on MDPs

#### the bellman equations
the value function, and the q function satisfy the following equations:
value function
$V^\pi(s) = \mathbb{E}_{\pi}[R(s,a) + \gamma V^\pi(s')]$
$V^\pi(s) = \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s,a) (R(s,a,s') + \gamma V^\pi(s'))$

q function
$Q^\pi(s,a) = \mathbb{E}_{\pi}[R(s,a) + \gamma V^\pi(s')]$
$Q^\pi(s,a) = \sum_{s' \in S} P(s'|s,a) (R(s,a,s') + \gamma \sum_{a' \in A(s')} \pi(a'|s') Q^\pi(s',a'))$

#### value iteration
attempts to find $V_{\pi^*}(S)$ using the following iterative process:

```python
def eval_state(state, V, S, A, P, R, gamma):
    v = max([sum([P[state][a][s_] * (R[state][a][s_] + gamma * V[s_]) for s_ in S]) for a in A[state]])

    return v


def value_iteration(S, A, P, R, gamma, theta=1e-6):
    '''
    finds the value function V for the optimal policy

    Args:
    S (list): set of states
    A (list): set of actions for state s
    P (dict): transition probability distribution
    R (dict): reward function
    gamma (float): discount factor
    theta (float): convergence threshold

    Returns:
    V (np.array): value matrix for the optimal policy
    '''
    V = np.zeros(len(S))
    delta = np.inf

    while delta > theta:
        delta = 0

        for s in S:
            v = V[s]
            V[s] = eval_state(s, V, S, A, P, R, gamma)
            delta = max(delta, abs(v - V[s]))

    return V
```

#### policy iteration
attempts to find $\pi^*(S)$ using the following iterative process:

```python
def policy_evaluation(policy, S, A, P, R, gamma, theta=1e-6):
    V = np.zeros(len(S))
    delta = np.inf

    while delta > theta:
        delta = 0

        for s in S:
            v = V[s]
            a = policy[s]
            V[s] = sum([P[s][a][s_] * (R[s][a][s_] + gamma * V[s_]) for s_ in S])
            delta = max(delta, abs(v - V[s]))

    return V


def policy_improvement(V, S, A, P, R, gamma):
    policy = {}

    for s in S:
        policy[s] = max(A[s], key=lambda a: sum([P[s][a][s_] * (R[s][a][s_] + gamma * V[s_]) for s_ in S]))

    return policy


def policy_iteration(S, A, P, R, gamma, theta=1e-6):
    '''
    Finds the optimal policy using policy iteration.

    Args:
    S (list): set of states
    A (list): set of actions for state s
    P (dict): transition probability distribution
    R (dict): reward function
    gamma (float): discount factor
    theta (float): convergence threshold

    Returns:
    policy (dict): optimal policy
    V (np.array): value function for the optimal policy
    '''
    policy = {s: A[s][0] for s in S}

    while True:
        V = policy_evaluation(policy, S, A, P, R, gamma, theta)
        new_policy = policy_improvement(V, S, A, P, R, gamma)

        # Check for convergence
        if new_policy == policy:
            break

        policy = new_policy

    return policy, V
```

### monte carlo
Monte Carlo methods are a class of algorithms that rely on repeated random sampling to obtain numerical results. They are widely used in reinforcement learning for estimating value functions and optimizing policies.

#### multi-armed bandits
The multi-armed bandit problem is a simplified reinforcement learning scenario where an agent must choose between multiple actions (or "arms") to maximize cumulative rewards. Each arm has an unknown reward distribution, and the agent must balance exploration (trying different arms to learn their rewards) and exploitation (choosing the arm with the highest known reward).

#### UCB (Upper Confidence Bound)
Upper Confidence Bound (UCB) is a popular algorithm for solving the multi-armed bandit problem. It balances exploration and exploitation by selecting actions based on both their estimated rewards and the uncertainty of those estimates. The UCB formula is typically expressed as:
$$
a_t = \arg\max_a \left( \hat{R}_a + c \sqrt{\frac{\ln t}{N_a}} \right)
$$
where:
- $\hat{R}_a$ is the estimated reward for action $a$,
- $N_a$ is the number of times action $a$ has been selected,
- $t$ is the current time step,
- $c$ is a tunable parameter controlling the exploration-exploitation tradeoff.

#### MCTS (Monte Carlo Tree Search)
Monte Carlo Tree Search (MCTS) is a search algorithm used for decision-making in environments with large or infinite state spaces, such as games. MCTS builds a search tree incrementally by simulating random trajectories and using the results to improve the tree's estimates. The algorithm consists of four main steps:
1. **Selection**: Traverse the tree from the root to a leaf node using a selection policy (e.g., UCB).
2. **Expansion**: Add one or more child nodes to the tree by exploring new actions.
3. **Simulation**: Perform a random simulation (rollout) from the newly added node to estimate the outcome.
4. **Backpropagation**: Update the values of the nodes along the path from the simulated node back to the root.

MCTS is widely used in applications like board games (e.g., Go, Chess) and planning problems, where it provides a balance between exploration and exploitation while handling large state spaces effectively.


## what is reinforcement learning
### basic definition
the reinforcement learning definition is an MDP with unknown P, R, and $\rho$.

### basic concepts
model-based, and model-free
model-based - in model-based reinforcement learning, the agent approximates the unknowns of the MDP (e.g., P, R, and $\rho$). after the approximation we can optimize a policy for the objective of the MDP.
model-free - in model-free reinforcement learning, the agent optimizes a policy without modeling the MDP.

offline, and online
offline - offline learning is a type of reinforcement learning where the agent learns from a fixed dataset of pre-existing experiences, without interacting with the environment.
online - online learning is where the agent learns by interacting with the environment.

off policy, and on policy
on policy - on policy is a learning process where only learn from experiance generated by the current policy
off policy - none off policy learning policy

exploration vs exploitation - this is a dilema thats about balancing between
exploration - trying new actions to discover their effects
exploitation - choosing actions that maximize known rewards.

sample efficiency - how efficiently we use the learning experiances we've gathered to learn

## deep reinforcement learning
value based methods - [DQN](https://nn.labml.ai/rl/dqn/index.html), Double DQN, and PER DQN
policy based - policy gradient, and [PPO](https://nn.labml.ai/rl/ppo/index.html)
actor critic - AC, A2C, A3C, and SAC
model based
imitation learning - behaviour cloneing, and DAGGER
monte carlo

common techniques
- n-step - A method that extends updates to include rewards from the next n steps, this gives updates with better look out.
- dueling/advantage - instead of computing $Q(a|s)$ directly we compute $Q(a|s)=V(s)+A(a|s)$ ,where $A(a|s)$ is the advantage function
- distributional - A method that models the distribution of possible rewards instead of just their expected value, providing richer information for decision-making, and forces the agent to learn more about the environment.

### A2C
**Advantage Actor-Critic (A2C)**
Initialize parameter vectors $\theta$ (actor), $\phi$ (critic), and learning rates $\alpha_\theta$, $\alpha_\phi$.
**for** each iteration **do**
  **for** each environment step **do**
    Sample action $\mathbf{a}_t \sim \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)$.
    Execute action $\mathbf{a}_t$ in the environment to observe reward $r_t$ and next state $\mathbf{s}_{t+1}$.
    Store transition $(\mathbf{s}_t, \mathbf{a}_t, r_t, \mathbf{s}_{t+1})$ in memory.
  **end for**

  **for** each gradient step **do**
    Compute the advantage estimate: $A_t = r_t + \gamma V_\phi(\mathbf{s}_{t+1}) - V_\phi(\mathbf{s}_t)$.
    Update the critic parameters: $\phi \leftarrow \phi - \alpha_\phi \nabla_\phi \left(A_t^2\right)$.
	Update the actor parameters: $\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t) A_t$.
  **end for**
**end for**

### SAC
**Soft Actor-Critic**
Initialize parameter vectors $\psi, \bar{\psi}, \theta, \phi$.
**for** each iteration **do**
  **for** each environment step **do**
    $\mathbf{a}_t \sim \pi_\phi(\mathbf{a}_t|\mathbf{s}_t)$
    $\mathbf{s}_{t+1} \sim p(\mathbf{s}_{t+1}|\mathbf{s}_t, \mathbf{a}_t)$
    $\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{s}_t, \mathbf{a}_t, r(\mathbf{s}_t, \mathbf{a}_t), \mathbf{s}_{t+1})\}$
  **end for**

  **for** each gradient step **do**
    $\psi \leftarrow \psi - \lambda_V \hat{\nabla}_\psi J_V(\psi)$
    $\theta_i \leftarrow \theta_i - \lambda_Q \hat{\nabla}_{\theta_i} J_Q(\theta_i)$ for $i \in \{1, 2\}$
    $\phi \leftarrow \phi - \lambda_\pi \hat{\nabla}_\phi J_\pi(\phi)$
    $\bar{\psi} \leftarrow \tau\psi + (1-\tau)\bar{\psi}$
  **end for**
**end for**

### BC
**Behavioral Cloning (BC)**
Initialize parameter vector $\theta$ for the policy $\pi_\theta$ and learning rate $\alpha$.
**Input**: Dataset $\mathcal{D} = \{(\mathbf{s}_i, \mathbf{a}_i)\}_{i=1}^N$ of state-action pairs.
**for** each gradient step **do**
	Sample a mini-batch of state-action pairs $\{(\mathbf{s}_j, \mathbf{a}_j)\}_{j=1}^M$ from $\mathcal{D}$.
	Compute the loss: $L(\theta) = \frac{1}{M} \sum_{j=1}^M HuberLoss(\pi_\theta(\mathbf{s}_j), \mathbf{a}_j)$.
	Update the policy parameters: $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$.
**end for**

## other considerations
when to use RL - use RL when you have a task with serial decision making, the task shouldn't have an optimal, and efficient solution.

what type of algorithm to use?

instability sources in RL

selecting hyperparameters
