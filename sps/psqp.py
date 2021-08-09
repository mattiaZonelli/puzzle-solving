import torch


def psqp(A, N, lr=1e-3):
    device = A.device
    active = torch.full((N, N), fill_value=True, device=device)
    p = torch.empty((N, N), device=device)

    for Na in range(N):
        p[active] = 1. / (N - Na)
        d = A @ p.flatten()
        p[active] = p[active] + lr * d
        p.clamp_(0., 1.)
        active = (p != 0.) | (p != 1.)
    
    return p
