import numpy as np
import torch

from exps.oracle import oracle_compatibilities
from sps.psqp import *

IT = 10  # hyperparameter


# utility function to init the matrix of probabilities - faster but brake the alg
def initBoard(dims):
    h, w = dims
    n = h * w
    p = torch.empty([n * n, 1])
    for i in range(n * n):
        p[i][0] = np.random.uniform(1 / n - 0.005, 1 / n + 0.005)  # slightly perturbated barycenter
    p = p.reshape(n, n)
    p3 = p / p.sum(dim=-1).unsqueeze(-1)
    return p3.reshape(n * n, 1)


# init prob matrix to barycenter - slower but working fine
def initBoard2(dims):
    h, w = dims
    n = h * w
    p = torch.empty([n, n])
    for i in range(n):
        for j in range(n):
            p[i][j] = 1/n
    return p.reshape(n * n, 1)


# core of the relaxation labeling
def relaxation_labelling(p, dims, order):
    h, w = dims
    SIDE = h * w
    Ch, Cv = oracle_compatibilities(h, w, order)

    rij = compatibilities(Ch, Cv, torch.tensor([h, w]))  # todo rinominare
    prev = 0
    diff = 1
    step = 0

    while diff > 0.001 and step < 200:
        q = torch.mm(rij, p)  # return dense vector
        numeratore = (p * q).reshape(SIDE, SIDE)
        if step <= IT:
            denominatore = numeratore.sum(dim=1, keepdims=True)
        else:
            if step % 2 == 0:
                denominatore = numeratore.sum(dim=0, keepdims=True)
            else:
                denominatore = numeratore.sum(dim=1, keepdims=True)

        #denominatore = numeratore.reshape(SIDE, SIDE).sum(dim=1).unsqueeze(1)

        # update values in the matrix of probabilities
        p = (numeratore / denominatore).reshape(SIDE * SIDE, 1)

        ''' for euclidian distance '''
        diff = torch.linalg.norm(p - prev)
        prev = p
        step += 1

    p[p < 1e-8] = 0.
    #p = p / p.reshape(SIDE, SIDE).sum(dim=1, keepdims=True)  # todo normalizzazione
    p = torch.nan_to_num(p)
    if step >= 200:
        print("STEP limit reached")

    return p


# look at the probability vector and assign to the cell the label with the highest value in the vector
def solve_puzzle(dims, order):
    p = relaxation_labelling(initBoard2(dims), dims, order)
    SIDE = dims[0] * dims[1]
    puzzle = torch.empty(SIDE, 1)
    for i in range(SIDE):
        pos = torch.argmax(p.reshape(SIDE, SIDE)[i])
        puzzle[i] = pos
    return puzzle.T

