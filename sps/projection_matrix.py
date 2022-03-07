import torch


def projection_matrix(p_vect, N):
    p = p_vect.reshape(N, N)  # if possible keep it stored as vector, do not reshape it as matrix
    N = len(p[0])

    ones = torch.ones(N, 1)
    P1 = torch.matmul(p, ones) + (torch.rand(N, 1) / 1e5)
    v1 = P1 - ones  # P1 = 1 -> P1-1=0

    Pt = torch.t(p)
    Pt1 = torch.matmul(Pt, ones) + (torch.rand(N, 1) / 1e5)
    v2 = Pt1 - ones  # Pt1 = 1 -> Pt1-1=0

    # all p_ij >= 0 --> produttoria per riga >= 0,
    # p_ij = torch.prod(p, 1).reshape(N, 1)
    p_ij = p_vect + (torch.rand(N**2, 1) / 1e3)  # non mi piace molto, provare con 1e5 o 1e2 o 1e3 e scegliere il migliore

    # n_i  = v_i / ||v_i||
    n1 = v1 / torch.linalg.norm(v1)
    n2 = v2 / torch.linalg.norm(v2)
    n3 = p_ij / torch.linalg.norm(p_ij)

    Nq = torch.cat([n1, n2, n3.reshape(N, N)], 1)

    '''Pq_mid = torch.matmul(torch.t(Nq), Nq)  # (NqT Nq)
    Pq_mid_inv = torch.inverse(Pq_mid)  # (NqT Nq)^-1
    # identity = torch.mm(Pq_mid, Pq_mid_inv).clamp_(0., 1.) # non una vera identity
    Pq_mid1 = torch.matmul(Nq, Pq_mid_inv)  # Nq (NqT Nq)^-1
    Pq = torch.matmul(Pq_mid1, torch.t(Nq))  # Nq (NqT Nq)^-1 NqT
    '''
    Pq = torch.mm(Nq, torch.linalg.pinv(Nq))  # with pseudo inverse

    return Pq.reshape(N ** 2, 1), Nq


def lambdas(p, Nk, N):
    p = p.reshape(N, N)
    NkTx = torch.mm(Nk.t(), p)
    bk = NkTx - (torch.rand(len(NkTx), len(NkTx[0])) / 1e3)  # forse potrebbe essere uguale a zero ???
    lambdas_x = NkTx - bk
    return lambdas_x