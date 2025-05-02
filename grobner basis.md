# Gröbner basis

## Applications

### solving SAT (NP-hard)
problem statement
given a a CNF logical clause $C_1\land ... \land C_n$ where $C_i$ are $\x_1\lor \lnot \x_2\lor \x_3\lor \x_4...$, we'd like to determine if it is satisfiable.

theorem
the CNF clause $C_1\land ... \land C_n$ is satisfiable if and only if the grobner basis of $\lbrace P_1, P_2, ..., P_n \rbrace$ isn't $\lbrace 1 \rbrace$.

proof - !!! TO DO !!!

### cryptanalysis
