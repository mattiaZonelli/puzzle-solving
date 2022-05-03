import torch.linalg

from sps.projection_matrix import *


# final example
def delta_f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    x3 = x[2][0]
    x4 = x[3][0]
    return torch.tensor([[2 * x1], [x2], [x3], [2 * x4]])


def constraints_matrix():
    Nk = torch.tensor([[1., 1., 0., 0.]]).T
    bk = torch.tensor([[1.]])
    Np = torch.tensor([[0., 0., 1., 1.], [1., 0., 1., 0.], [0., 1., 0., 1.]]).T
    bp = torch.tensor([[1., 1., 1.]])

    # Np = torch.tensor([[]])
    # bp = torch.tensor([[]])
    # Nk = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.], [1., 0., 1., 0.], [0., 1., 0., 1.]]).T
    # bk = torch.tensor([[1., 1., 1., 1.]])

    return Np, Nk, bp, bk


# example from youtube
'''def delta_f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    x3 = x[2][0]
    x4 = x[3][0]
    return torch.tensor([[-2 * x1 + 2], [-2 * x2], [-2 * x3], [-2 * x4 + 3]])


def constraints_matrix():
    Np = torch.tensor([[],
                       [],
                       [],
                       []])
    bp = torch.tensor([[]])
    Nk = torch.tensor([[-2., -1., 0., -2.],
                       [-1., -1., 1., 0.],
                       [-1., -2., 1., 0.],
                       [-4., -1., 0., -4.]])
    bk = torch.tensor([[-7., -2.2, 0., -7.]])
    # con la normalizzazione dei constraints sembra non funzionare proprio
    # Np = Np / torch.linalg.norm(Np, dim=0)
    # Nk = Nk / torch.linalg.norm(Nk, dim=0)

    return Np, Nk, bp, bk'''

# example from other paper
'''def delta_f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    return torch.tensor([[-4*x1 + 2*x2 + 4], [-4*x2 + 2*x1 + 6]])


# example from other paper
def constraints_matrix():
    Np = torch.tensor([[-1.],
                       [-5.]])
    bp = torch.tensor([[-5.]])
    Nk = torch.tensor([[-1., -1., 0.],
                       [-1., 0., -5.]])
    bk = torch.tensor([[-1.9032, -1.129, -3.871]])
    return Np, Nk, bp, bk'''

# example from Rao
'''def delta_f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    return torch.tensor([[-2*x1 + 2], [-2*x2 + 4]])


def constraints_matrix():
    Np = torch.tensor([[-1.],
                       [-4.]])
    bp = torch.tensor([[-4.9999]])
    Nk = torch.tensor([[-2., -1., 0.],
                       [-3., 0., -4.]])
    bk = torch.tensor([[-4.7058, -0.7647, -4.2352]])
    return Np, Nk, bp, bk'''


def init_ProjectionMatrix(Np, Nk, bp, bk, N):
    n1 = torch.index_select(Nk, 1, torch.tensor(0))
    Pq = torch.eye(N, N) - (n1 @ n1.T)
    Nk, Np, bk, bp = move_constraint(torch.tensor(0), Nk, Np, bk, bp)
    inverse = torch.inverse(n1.T @ n1)
    return Pq, inverse, Nk, Np, bk, bp


def update_Pq(q, Pq, inverse, Nk):
    t = torch.tensor([])
    for i in q:
        nq = torch.index_select(Nk, 1, i)
        rq_1 = torch.inverse(Nk.T @ Nk) @ (Nk.T @ nq)
        Pq_1nq = nq - (Nk @ rq_1)
        if len(torch.nonzero(Pq_1nq)) > 0:
            A0 = torch.pow(torch.linalg.norm(Pq_1nq), 2)
            B4 = 1. / A0
            B2 = (-1.) * (1. / A0) @ rq_1
            B3 = B2.T
            B1 = inverse + (1. / A0) * (rq_1 @ rq_1.T)
            inverse = torch.tensor([[B1, B2], [B3, B4]])
            uq = Pq_1nq / torch.linalg.norm(Pq_1nq)
            Pq -= (uq @ uq.T)
            t = torch.cat([t, i], 0)
    return Pq, inverse, t


def raoAlg(x):
    Np, Nk, bp, bk = constraints_matrix()

    Pq, inverse, Nk, Np, bk, bp = init_ProjectionMatrix(Np, Nk, bp, bk, x.shape[0])

    for i in range(100):
        d = delta_f(x)  # gv

        s = (Pq @ d)

        if torch.linalg.norm(s) <= 1e-3:  # S == 0
            # in rosen's paper is called r
            lambdas = ((inverse @ Np.T) @ d)
            if torch.max(lambdas) <= 1e-4:  # if ALL lambdas are <= 0
                return x
            else:  # some components of lambdas are positive
                # drop Hq corresponding to rq = max{ri} > 0
                q = torch.argmax(lambdas).unsqueeze(0)
                Pq, inverse, q = update_Pq(q, Pq, inverse, Nk)
                Np, Nk, bp, bk = move_constraint(q, Np, Nk, bp, bk)  # drop q-th constraint_ from Np
        else:
            # normalize s with norma max first, then whit norm
            s /= torch.linalg.norm(s, float('inf'))
            s /= torch.linalg.norm(s)
            lambda_M, M = steplength(x, Nk, bk, s)
            # if lambda_M < 1e-4:
            # print(" BISOGNA AGIRE SULLA STEPLENGTH")

            d_lM = (s.T @ delta_f(x + lambda_M * s))  # z T g'_v+1 (Rosen)
            if d_lM < 1e-4:  # d_lM < 0
                # interpolation as in Rosen
                zTg = (s.T @ d)
                rho = zTg / (zTg - d_lM)
                # rho = rho.clamp_(0., 1.)
                x = (rho * (x + lambda_M * s)) + ((1 - rho) * x)
                # print("rho", x)
            else:  # d_lM >= 0
                if len(M) > 1:
                    # print("check z da fare")
                    M = check_z(s, Nk)
                if M.nelement() == 0:
                    # print(" nessun elemento trovato in check_z")
                    lambda_M = 1.
                    ##x += 1. * s
                    ##return x
                else:
                    M = M.unsqueeze(0) if M.ndim == 0 else M
                    Pq, inverse, q = update_Pq(M, Pq, inverse, Nk)
                    if q.nelement() != 0:
                        Nk, Np, bk, bp = move_constraint(q, Nk, Np, bk, bp)  # add the q-th constraint to Np
                x += lambda_M * s

    print("fine range(100)")
    return x


if __name__ == '__main__':
    # example from youtube
    '''x = torch.tensor([[0.], [0.], [0.], [0.]])
    x1 = torch.tensor([[0.9], [0.], [0.], [1.3]])'''
    # example from other paper
    '''x = torch.tensor([[0.], [1.]])
    x1 = torch.tensor([[1.129], [0.7742]])'''
    # example from Rao
    '''x = torch.tensor([[0.7647], [1.0588]])
    x1 = torch.tensor([[4.9999], [0]])'''
    # final example
    x = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    x1 = torch.tensor([[0.25], [0.25], [0.25], [0.25]])

    print(raoAlg(x))
    # print(raoAlg(x1))
