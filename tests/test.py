import torch.linalg
from scipy.optimize import line_search
from sps.projection_matrix import *


# final example
'''def delta_f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    x3 = x[2][0]
    x4 = x[3][0]
    return torch.tensor([[2 * x1], [x2], [x3], [2 * x4]])


def constraints_matrix():
    # Nk = torch.tensor([[]])
    # bk = torch.tensor([[]])
    # Np = torch.tensor([[0., 0., 1., 1.], [1., 0., 1., 0.], [0., 1., 0., 1.]]).T
    # bp = torch.tensor([[1., 1., 1.]])

    Np = torch.tensor([[-1., -1., 0., 0.]]).T
    bp = torch.tensor([[-1.]])
    Nk = torch.tensor([[0., 0., -1., -1.], [-1., 0., -1., 0.], [0., -1., 0., -1.]]).T
    bk = torch.tensor([[-1., -1., -1.]])

    return Np, Nk, bp, bk'''


# example from youtube
'''def delta_f(x):
    x1 = x[0][0]
    x2 = x[1][0]
    x3 = x[2][0]
    x4 = x[3][0]
    return torch.tensor([[-2 * x1 + 2], [-2 * x2], [-2 * x3], [-2 * x4 + 3]])


def constraints_matrix():
    Np = torch.tensor([[-2.],
                       [-1.],
                       [-1.],
                       [-4.]])
    bp = torch.tensor([[-7.]])
    Nk = torch.tensor([[-1., 0., -2., -1.],
                       [-1., -1., 0., 0.],
                       [-2., -1., 0., -1.],
                       [-1., 0., -4., -1.]])
    bk = torch.tensor([[-2.2, 0., -7., -2.2]])
    # con la normalizzazione dei constraints sembra non funzionare proprio
    # Np = Np / torch.linalg.norm(Np, dim=0)
    # Nk = Nk / torch.linalg.norm(Nk, dim=0)

    return Np, Nk, bp, bk'''

# example from other paper
def delta_f(x):
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
    return Np, Nk, bp, bk

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


def raoAlg(x):
    Np, Nk, bp, bk = constraints_matrix()

    for i in range(15):
        d = delta_f(x)  # gv

        Pq = projection_matrix(x.shape[0], Np)

        s = (Pq @ d)

        if -1e-4 <= torch.linalg.norm(s) <= 1e-4:  # S == 0, non si può andare più giù di e-4
        #if torch.linalg.norm(s) <= 1e-3:
            # in rosen's paper is called r
            lambdas = d if Np.nelement() == 0 else ((torch.inverse(Np.T @ Np) @ Np.T) @ d)
            if torch.max(lambdas) <= 1e-4:  # if ALL lambdas are <= 0
                return x
            else:  # some components of lambdas are positive
                # drop Hq corresponding to rq = max{ri} > 0
                q = torch.argmax(lambdas).unsqueeze(0)
                Np, Nk, bp, bk = move_constraint(q, Np, Nk, bp, bk)  # drop q-th constraint_ from Np

        else:
            # normalize s with norma max first, then whit norm
            s /= torch.linalg.norm(s, float('inf'))
            s /= torch.linalg.norm(s)
            lambda_M, M = steplength(x, Nk, bk, s, torch.full((x.shape[0], 1), fill_value=True))

            '''def obj_foo(x):
                x1 = x[0]
                x2 = x[1]
                return torch.tensor([2*x1**2 + 2*x2**2 - 2*x1*x2 - 4*x1 - 6*x2]).squeeze()
            def obj_grad(x):
                x1 = x[0]
                x2 = x[1]
                return torch.tensor([[4*x1 - 2*x2 - 4], [4*x2 - 2*x1 - 6]]).squeeze()
            step = line_search(obj_foo, obj_grad, x.squeeze(), -s.squeeze())
            lambda_M = torch.tensor(step[0]) if step[0] != None else torch.tensor(0.)

            x+= lambda_M*s'''

            d_lM = (s.T @ delta_f(x + lambda_M * s))  # z T g'_v+1 (Rosen)
            if d_lM < 0.:  # d_lM < 0
                # interpolation as in Rosen
                zTg = (s.T @ d)
                rho = zTg / (zTg - d_lM)
                # rho = rho.clamp_(0., 1.)  # rho dovrebbe essere compreso tra 0 e 1
                x = (rho * (x + lambda_M * s)) + ((1 - rho) * x)
                # print("rho", x)
            else:  # d_lM >= 0
                if len(M) > 1:
                    M = check_z(s, Nk)
                if M.nelement() != 0:
                    Nk, Np, bk, bp = move_constraint(M, Nk, Np, bk, bp)  # add the q-th constraint to Np
                x += lambda_M * s

    print("fine range")
    return x


if __name__ == '__main__':
    # example from youtube
    '''x = torch.tensor([[2.], [2.], [1.], [0.]])
    x1 = torch.tensor([[0.9], [0.], [0.], [1.3]])'''
    # example from other paper
    x = torch.tensor([[0.], [1.]])
    x1 = torch.tensor([[1.129], [0.7742]])
    # example from Rao
    '''x = torch.tensor([[1.], [1.]])
    x1 = torch.tensor([[4.9999], [0]])'''
    # final example
    '''x = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    x1 = torch.tensor([[0.25], [0.25], [0.25], [0.25]])'''

    print(raoAlg(x).T)
    print(raoAlg(x1).T)



'''
    BISOGNA AGGIUNGERE RESTORATION MOVESS?
'''
