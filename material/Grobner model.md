# Grobner model

## Problem statement

Buchberger's algorithm which is used to computeing the Gröbner basis of an ideal has an arbitrary selection of which pair of polynomials to reduce, this selection has a large impact on the efficiency of the algorithm due to the fact fact that many pairs are reduced to zero (no advancement in the algorithm).
we'd like to create a model which will choose the best pair of polynomials to reduce at each step of the algorithm to minimize the number of steps needed to compute the Gröbner basis.

input - the generators of the ideal, and the reducible pairs of polynomials
output - evaluation of the pair of polynomials to reduce

example of input
$p_0(x)=x_0^2 x_2^2 + 13466 x_0^3$
$p_1(x)=x_0 x_1 x_2^2 + 17385 x_1^2 x_2^2$
$p_2(x)=x_0^2 x_1 + 17034 x_2$
$p_3(x)=x_0 x_1 x_2^2 + 5600 x_0^2 x_2$

where $p_i(x_0, x_1, x_2) \in \mathbb{Z}_{32003}[x_0, x_1, x_2]$ are the generators of the ideal $I = \langle p_0, p_1, p_2, p_3 \rangle$.

with reducible pairs
(0, 1)
(0, 2)
(1, 3)

utility function - the number of steps needed to compute the Gröbner basis.

## data

the input to the neural network will be

- the generators of the ideal - $\{\{m_{11},..., m_{1n_1}\}, \{m_{21},..., m_{2n_2}\}, ..., \{m_{k1},..., m_{kn_k}\}\}$ where $m_{ij}$ is the $j$-th monomial of the $i$-th polynomial, and $n_i$ is the number of monomials in the $i$-th polynomial.
- a mask indicating which polynomials weren't already reduced to zero or were already reduced

### data representation


the ideals used to do [Algebraic Cryptanalysis Scheme of AES-256](https://onlinelibrary.wiley.com/doi/pdf/10.1155/2017/9828967) have 720 generators, of degree 254, with each monomial having a maximum of 10 variables.
to store such ideals we'll use $2*10*254*720=3657600$ bytes, to store a batch of size

### Environment

we'll use an adaptation of sympy's implementation of Buchberger's algorithm with our model as the selector of which pair of polynomials to reduce.

the ideals will start simple, and we'll increase the task complexity (more generators, longer polynomials, and more variables) if the model improvement plateaus.

the reward would be the difference in a consistent heuristic.

## Model structure

here are a few considerations for the model

1. there should be respect to symetries of the input, the symetries are the order of the polynomials, monomials.
2. the model should be able to accept a set of polynomials without restrictions on the number of polynomials, variables or the degree of the polynomials

model structure

1. representation of the ideal as a sparse array
2. variable embedding
3. monomial embedding
4. polynomial embedding
5. get end product (q values, likelihoods, and etc.)

## Training

due to the long term nature of the problem, we'll use a reinforcement learning approach to train the model.

here are the methods we'll consider for training the model

1. [DQN](https://proceedings.neurips.cc/paper/2016/file/8d8818c8e140c64c743113f563cf750f-Paper.pdf) with a subset of the following improvements double DQN, dueling DQN, PER, noisy networks, distributional DQN, and multi-step learning. using all of the improvements is called [Rainbow](https://arxiv.org/pdf/1710.02298).
2. [PPO](https://arxiv.org/pdf/1707.06347)
3. [A3C](https://arxiv.org/pdf/1602.01783)

### optimization

the optimization algorithm will be Adam, with gradient clipping, learning rate scheduler, and batch size between 256 and 16384.

### callbacks

we'll use the following callbacks checkpoints, evaluation, and early stopping.

### hardware

we'll use an 8 a100 gpu server

## Evaluation

the evaluation will compare our adapteation of Buchberger's algorithm with the following algorithms - vanilla Buchberger's algorithm, improved Buchberger's algorithm, F4, and F5.
