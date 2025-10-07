# project proposal - improving algebraic cryptanalysis using Reinforcement Learning
## introduction
Many difficult computational problems, including those in cryptography, can be translated into a set of polynomial equations over a finite field. Solving these systems of equations can be done using a powerful algebraic tool called a Gröbner basis. The standard method for this, Buchberger's algorithm, has a critical weakness: its performance is highly dependent on the order of its computational steps.

Previous work has shown that reinforcement learning can learn to choose these steps intelligently, outperforming standard heuristics on smaller problems. This project aims to build on that success.

We will explore advanced reinforcement learning techniques—such as imitation learning, curriculum learning, and AlphaZero variants—to tackle much larger and more complex systems of polynomial equations. The ultimate goal is to solve problems that are currently considered infeasible, with potential applications in improving algebraic cryptanalysis methods.

more formally one can represent every problem in PSPACE (and also NP) as a membership problem of a polynomial ideal over the field $\mathbb{F}_2$ $I = \langle f_1, f_2, ..., f_m \rangle$ where $f_i \in \mathbb{F}_2[x_1, x_2, ..., x_n]$. this membership problem could be solved using Gröebner basis, and there are which shown that using Gröebner basis can outperform SAT solvers ([yaac](https://gitlab.com/upbqdn/yaac.git)). however, the time for computing a Gröebner basis is highly dependent on the order in which the reduction steps of the Buchberger's algorithm are done.

there is a previous [work](https://github.com/dylanpeifer/deepgroebner/tree/master) which shown that for small and simple systems of polynomial equations, one can use reinforcement learning to learn which reduction step to do next, and outperform the standard heuristics.

in this project, we various methods from reinforcement learning to improve the performance of Buchberger's algorithm, and try to solve systems of polynomial equations that are currently infeasible. the methods we will try include imitation learning, deep reinforcement learning, curriculum learning, and variants of AlphaZero.

## for candidates
### what you will do
here is the list of research stages, not all will be done and you will have some choice in what to do:
- learn the essentials - solving polynomial systems, [Gröebner basis](http://www.scholarpedia.org/article/Groebner_basis), [Buchberger's algorithm](http://www.scholarpedia.org/article/Buchberger%27s_algorithm) getting familiar with previous [work](https://gitlab.com/upbqdn/yaac.git)
- get familiar with the code base - [groebnerRL](https://github.com/IlayMenahem/grobnerRl.git)
- system of polynomial equations generation - use [yaac](https://gitlab.com/upbqdn/yaac.git) to generate systems of polynomial equations of varying number of variables, equations, and difficulty.
- imitation learning - create an agent that uses MCTS to create a dataset of good actions, and train a neural network to imitate it. for this task the code is almost done, it's only necessary to train a few models and evaluate them.
- reinforcement learning - take the agent that imitates MCTS and improve it using reinforcement learning, [AlphaZero](https://arxiv.org/abs/1712.01815), or [Gumbel AlphaZero](https://github.com/grimmlab/policy-based-self-competition.git).
- curriculum learning - add a curriculum learning scheme to the reinforcement learning part, so the agent maybe able to solve problems that are currently infeasible.

### expected background
required:
- python
- basic knowledge in pytorch
- linear algebra

non of the following is required, but the more you know the better:
- algebra - fields, polynomial rings, Gröbner basis, and Buchberger's algorithm
- complexity theory - the polynomial hierarchy, and understanding of polynomial reductions
- deep learning - transformers, imitation learning, reinforcement learning, MCTS, AlphaZero, and concurrent learning
