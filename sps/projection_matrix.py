import torch

# con ricorsione
'''def constraints_matrix2(N):
    one_hot = torch.eye(N, N)
    unos = torch.ones((N, 1))
    ones = torch.kron(one_hot, unos)
    ones1 = torch.kron(unos, one_hot)
    Np = torch.cat([ones, ones1], 1)  # active
    Nk = torch.eye(N ** 2, N ** 2)  # non-active

    bp = torch.cat([unos.T, unos.T], 1)
    bk = torch.zeros((1, N ** 2))
    return Np, Nk, bp, bk


def projection_matrix2(i, q, Np, N):
    if i < 0:
        ValueError(f"C'è un bel errore")
    if i == 0:
        I = torch.eye(N ** 2, N ** 2)
        n1 = torch.index_select(Np, 1, torch.tensor(i))
        n1 = n1 / torch.linalg.norm(n1)
        P1 = I - (n1 @ n1.T)
        return P1, torch.inverse(n1.T @ n1), n1
    else:
        Pq_1, inverted, Nq_1 = projection_matrix2(i-1, q, Np, N)
        nq = torch.index_select(Np, 1, torch.tensor(i))
        rq_1 = inverted @ (Nq_1.T @ nq)
        Pq_1_nq = nq - (Nq_1 @ rq_1)
        if torch.linalg.norm(Pq_1_nq) == 0:
            return projection_matrix2(i - 1, q, Np, N)
        else:
            A0 = torch.pow(torch.linalg.norm(Pq_1_nq), 2)
            B4 = torch.tensor([1 / A0])
            B2 = -1 * (B4 @ rq_1)
            B3 = B2.T
            B1 = inverted + (B4 @ rq_1 @ rq_1.T)
            uq = Pq_1_nq / torch.linalg.norm(Pq_1_nq)
            Pq = Pq_1 - (uq @ uq.T)
            return Pq, torch.tensor([[B1, B2], [B3, B4]]), nq
'''


# lin. indep constraints have been selected manually, should be automated?
def constraints_matrix(N: int) -> [torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    one_hot = torch.eye(N, N)
    unos = torch.ones((N, 1))
    ones = torch.kron(one_hot, unos)
    ones1 = torch.kron(unos, one_hot)
    # ni = ones / torch.linalg.norm(ones)  # ni need to be normalized ????
    Nk = torch.tensor([[]])
    Np = torch.cat([ones, ones1], 1)
    bk = torch.tensor([[]])
    bp = torch.cat([unos.T, unos.T], 1)
    return Np, Nk, bp, bk
    #return ones, ones1, unos.T, unos.T


def projection_matrix(N: int, Nq: torch.tensor) -> torch.tensor:
    mid = torch.inverse((Nq.T @ Nq))
    if torch.linalg.cond(mid) > 10.:
        ValueError(f"Booh che ne so")
    right = (mid @ Nq.t())
    Pq = (Nq @ right)
    I = torch.eye(N, N)
    Pq = I - Pq
    return Pq


# move constraints with index in q, from from_N and from_b to to_N and to_b
def move_constraint(q: torch.tensor, from_N: torch.tensor, to_N: torch.tensor, from_b: torch.tensor, to_b: torch.tensor) \
        -> [torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    if to_N.nelement() == 0:
        to_N = torch.index_select(from_N, 1, q)
        to_b = torch.index_select(from_b, 1, q)
    else:
        add_to_to_b = torch.index_select(from_b, 1, q)
        to_b = torch.cat([to_b, add_to_to_b], 1)
        add_to_to_N = torch.index_select(from_N, 1, q)
        to_N = torch.cat([to_N, add_to_to_N], 1)

    indices = torch.ones(from_b.shape[1], dtype=bool)
    indices[q] = False
    from_b = from_b[:, indices]
    from_N = from_N[:, indices]

    return from_N, to_N, from_b, to_b


# calcolo lambda_M come Rao, ma la uso come steplenght come fa Rosen
def steplength(p: torch.tensor, Nk: torch.tensor, bk: torch.tensor, s: torch.tensor, active) \
        -> [torch.tensor, torch.tensor]:
    gk = (p.T @ Nk)
    gk = gk - bk
    denom = (s.T @ Nk)
    denom = torch.where(torch.abs(denom - 0.0) > 0.0001, denom, torch.zeros(denom.shape))
    lk = gk / denom
    lk = lk.abs()
    if len(lk[lk > 1e-4]) == 0 or len(lk.isfinite().nonzero()) == 0:
        lM = 0.5
        while (p[active] + lM * s[active]).max() < 0.9999:  # and (p[active] + lM * s[active]).min() > 0.:
            lM += 0.005
        return torch.tensor(lM), torch.tensor([])

    lM = (lk[lk > 1e-4]).min()
    # to get index of min(lk) > 0
    i = torch.where(torch.abs(lk - lM) < 1e-6)[1]
    return lM, i


def check_z(z: torch.tensor, Ni: torch.tensor) -> torch.tensor:
    thetas = z.T @ Ni
    thetas = torch.where(torch.abs(thetas - 0.0) > 0.0001, thetas, torch.zeros(thetas.shape))
    if thetas.nonzero().shape[0] == 0:  # funziona solo se non ci sono valori come 1e-4e etcc
        return torch.tensor([[]])
    theta_i = torch.min(thetas[thetas != 0.])  # da rivedere la condizione in modo da non aver problemi coi float
    i = torch.where(torch.abs(thetas - theta_i) < 1e-6)[1]
    return i[0]


def adjust_dependency(s, Np, Nk, bp, bk, N, d):
    s /= torch.linalg.norm(s, float('inf'))
    s /= torch.linalg.norm(s)  # z
    i = check_z(s, Nk)
    while i.numel() > 0:
        Nk, Np, bk, bp = move_constraint(i, Nk, Np, bk, bp)
        Pq = projection_matrix(N, Np)
        s = Pq @ d
        s /= torch.linalg.norm(s, float('inf'))
        s /= torch.linalg.norm(s)  # z
        i = check_z(s, Nk)
    return s, Np, Nk, bp, bk
