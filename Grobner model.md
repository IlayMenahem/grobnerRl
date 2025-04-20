Gröbner model

# Problem statment
Buchberger's algorithm which is used to computeing the Gröbner basis of an ideal has an arbitrary selection of which pair of polynomials to reduce, this selection has a large impact on the efficiency of the algorithm due to the fact fact that many pairs are reduced to zero (no advancement in the algorithm).
we'd like to create a model which will choose the best pair of polynomials to reduce at each step of the algorithm to minimize the number of steps needed to compute the Gröbner basis.

input - the generators of the ideal
output - $\{i,j\}$ $i\neq j$ the indices of polynomials to reduce

utility function - the number of steps needed to compute the Gröbner basis.

# data
the input to the neural network will be
- the generators of the ideal
- a mask indicating which polynomials are already reduced to zero, or were already reduced
- the field of the coefficients

## data representation
this data representation implies that we have a limited number of variables
1. finitely generated ideals - the set of polynomials that generate the ideal
2. polynomial - a set of tuples (coefficient, monomial)
3. monomial - a sparse vector of the power (int16) of each variable

the ideals used to do [Algebraic Cryptanalysis Scheme of AES-256](https://onlinelibrary.wiley.com/doi/pdf/10.1155/2017/9828967) have 720 generators, of degree 254, with each monomial having a maximum of 10 variables.
to store such ideals we'll use $2*10*254*720=3657600$ bytes, to store a batch of size

## Environment
we'll use an adaptation of sympy's implementation of Buchberger's algorithm with our model as the selector of which pair of polynomials to reduce.

the ideals will start simple, and we'll increase the task complexity (more generators, longer polynomials, and more variables) if the model improvement plateaus.

the reward would be the difference in a consistent heuristic.

# Model structure
here are a few considerations for the model
1. invariance to the order of the polynomials in the input set
2. the naming of the variables shouldn't change the output
3. the model should be able to accept a set of polynomials without restrictions on the number of polynomials, variables or the degree of the polynomials

model structure
1. representation of the ideal as a list[Polynomial] ,where Polynomial is list[tuple[coefficient, Monomial]], and Monomial is list[tuple[variable, exponent]]
2. variable embedding
3. monomial embedding
4. polynomial embedding
5. get end product (q values, likelihoods, and etc.)

# Training
due to the long term nature of the problem, we'll use a reinforcement learning approach to train the model.

here are the methods we'll consider for training the model
1. [DQN](https://proceedings.neurips.cc/paper/2016/file/8d8818c8e140c64c743113f563cf750f-Paper.pdf) with a subset of the following improvements double DQN, dueling DQN, PER, noisy networks, distributional DQN, and multi-step learning. using all of the improvements is called [Rainbow](https://arxiv.org/pdf/1710.02298).
2. [PPO](https://arxiv.org/pdf/1707.06347)
3. [A3C](https://arxiv.org/pdf/1602.01783)

## optimization
the optimization algorithm will be Adam, with initial learning rate ---TBA---, and batch size between 256 and 16384; we won't use a learning rate scheduler, or gradient clipping.

## callbacks
we'll use the following callbacks checkpoints, evaluation, and early stopping.

## hardware
we'll use an 8 a100 gpu server

# Evaluation
the evaluation will compare our adapteation of Buchberger's algorithm with the following algorithms - vanilla Buchberger's algorithm, improved Buchberger's algorithm, F4, and F5.
