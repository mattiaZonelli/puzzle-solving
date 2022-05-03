import numpy as np

from exps.oracle import oracle_compatibilities
from sps.psqp import *

IT = 10  # hyperparameter

# utility function to init the matrix of probabilities
def initBoard(dims):
    h, w = dims
    n = h * w
    p = torch.empty([n * n, 1])
    for i in range(n * n):
        p[i][0] = np.random.uniform(1 / n - 0.005, 1 / n + 0.005)
    p = p.reshape(n, n)
    p3 = p / p.sum(dim=-1).unsqueeze(-1)
    return p3.reshape(n * n, 1)


# core of the relaxation labeling
def relaxation_labelling(p, dims, order):
    h, w = dims
    SIDE = h * w
    Ch, Cv = oracle_compatibilities(h, w, order)

    rij = compatibilities(Ch, Cv, torch.tensor([h, w])).to_dense()
    prev = 0
    diff = 1
    step = 0

    while diff > 0.001 and step < 200:
        q = rij @ p
        numeratore = p * q
        if step <= IT:
            denominatore = numeratore.reshape(SIDE, SIDE).sum(dim=-1).unsqueeze(1)
        else:
            if step % 2 == 0:
                denominatore = numeratore.reshape(SIDE, SIDE).sum(dim=0).unsqueeze(0)
            else:
                denominatore = numeratore.reshape(SIDE, SIDE).sum(dim=-1).unsqueeze(1)

        # update values in the matrix of probabilities
        p = (numeratore.reshape(SIDE, SIDE) / denominatore).reshape(SIDE * SIDE, 1)

        ''' for euclidian distance '''
        diff = torch.linalg.norm(p - prev)
        prev = p
        step += 1

    p[p < 1e-4] = 0.
    p = torch.nan_to_num(p)

    return p


# look at the probability vector and assign to the cell the label with the highest value in the vector
def solve_puzzle(dims, order):
    p = relaxation_labelling(initBoard(dims), dims, order)
    SIDE = dims[0] * dims[1]
    puzzle = torch.empty(SIDE, 1)
    for i in range(SIDE):
        pos = torch.argmax(p.reshape(SIDE, SIDE)[i])
        puzzle[i] = pos
    return puzzle.T

