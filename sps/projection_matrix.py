import torch


def projection_matrix(p):
    N = len(p[0])

    ones = torch.ones(N, 1)
    P1 = torch.matmul(p, ones)
    v1 = P1 - ones  # P1 = 1 -> P1-1=0

    Pt = torch.t(p)
    Pt1 = torch.matmul(Pt, ones)
    v2 = Pt1 - ones  # Pt1 = 1 -> Pt1-1=0

    # all p_ij >= 0
    p_ij = P1

    # n_i  = v_i / ||v_i||
    n1 = v1 / torch.linalg.norm(v1)
    n2 = v2 / torch.linalg.norm(v2)
    n3 = p_ij / torch.linalg.norm(p_ij)

    Nq = torch.cat([n1, n2, n3], 1)

    Pq_mid = torch.matmul(torch.t(Nq), Nq)  # (NqT Nq) but is singular so...
    # print('Pq_mid\n', Pq_mid, '\n')
    # with small perturbations to make Pq_mid non-singular...
    Pq_mid = Pq_mid + (torch.rand(len(Pq_mid[0]), len(Pq_mid[0])) / 1e5)
    # print('Pq_mid perturbed\n', Pq_mid, '\n')
    # with pseudo inverse
    # Pq_mid_inv = torch.linalg.pinv(Pq_mid)  # (NqT Nq)^-1
    Pq_mid_inv = torch.inverse(Pq_mid)
    # print('Pq_mid_inv\n', Pq_mid_inv, '\n')
    # print('identity\n', torch.matmul(Pq_mid, Pq_mid_inv), '\n')
    Pq_mid1 = torch.matmul(Nq, Pq_mid_inv)  # Nq (NqT Nq)^-1
    Pq = torch.matmul(Pq_mid1, torch.t(Nq))  # Nq (NqT Nq)^-1 NqT

    return Pq
