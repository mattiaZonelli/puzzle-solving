import torch


# t tensor (n. tiles, color channels, pixel, pixel)... data['puzzle'].squeeze()
def compatibility_Ch(t):
    pairs = torch.combinations(torch.arange(t.shape[0]))
    diss = torch.zeros((t.shape[0], t.shape[0]))
    # parametri usati da Pomeranz et al. vedi PSQP
    p = 3 / 10
    q = 1 / 16
    # parametri usati da Cho et al. vedi PSQP
    # p = 2
    # q = 2
    for i, j in pairs:
        T = t.shape[2]
        for k in range(T):
            for l in range(3):
                diss[i][j] += torch.pow(torch.abs(t[i, l, k, T - 1] - t[j, l, k, 0]), p)
    diss = torch.pow(diss, (q / p))

    sigmas = torch.zeros(t.shape[0])
    for i in range(t.shape[0] - 1):
        sigmas[i] = diss[i, diss[i] > 0].min()
    sigmas = 2 * torch.pow(sigmas, 2)

    W = (- diss / sigmas.unsqueeze(1) + 1e-6).exp()
    W = torch.nan_to_num(W, 0.)
    C = (1. - torch.eye(*W.shape, device=t.device)) * (W > 0.)
    W = W * C
    W[W > 1 - 1e-4] = 0.
    vals, inds = W.topk(1, dim=1)
    W = torch.zeros_like(W)
    W[torch.arange(W.shape[0], device=W.device).repeat_interleave(1),
      inds.flatten()] = vals.flatten()
    W = torch.ones([W.shape[0], W.shape[1]]) - W
    W[W == 1.] = 0.
    return W


# t tensor (n. tiles, color channels, pixel, pixel)... data['puzzle'].squeeze()
def compatibility_Cv(t):
    pairs = torch.combinations(torch.arange(t.shape[0]))
    diss = torch.zeros((t.shape[0], t.shape[0]))
    # parametri usati da Pomeranz et al. vedi PSQP
    p = 3/10
    q = 1/16
    # parametri usati da Cho et al. vedi PSQP
    # p = 2
    # q = 2
    for i, j in pairs:
        T = t.shape[2]
        for k in range(T):
            for l in range(3):
                diss[i][j] += torch.pow(torch.abs(t[i, l, T - 1, k] - t[j, l, 0, k]), p)
    diss = torch.pow(diss, (q / p))

    sigmas = torch.zeros(t.shape[0])
    for i in range(t.shape[0] - 1):
        sigmas[i] = diss[i, diss[i] > 0].min()
    sigmas = 2 * torch.pow(sigmas, 2)

    W = (- diss / sigmas.unsqueeze(1) + 1e-6).exp()
    W = torch.nan_to_num(W, 0.)
    C = (1. - torch.eye(*W.shape, device=t.device)) * (W > 0.)
    W = W * C
    W[W > 1 - 1e-4] = 0.
    vals, inds = W.topk(1, dim=1)
    W = torch.zeros_like(W)
    W[torch.arange(W.shape[0], device=W.device).repeat_interleave(1),
      inds.flatten()] = vals.flatten()
    W = torch.ones([W.shape[0], W.shape[1]]) - W
    W[W == 1.] = 0.
    return W


# Le accuratezze non sono alte come nei paper ma abbastanza buone
def mgc(x):
    n_tiles = x.shape[0]
    C_LR = torch.zeros((n_tiles, n_tiles))
    D_LR = torch.zeros((n_tiles, n_tiles))
    D_RL = torch.zeros((n_tiles, n_tiles))
    for i in range(n_tiles):
        for j in range(n_tiles):
            P = x.shape[2]
            G_iL = (x[i, :, :, P - 1] - x[i, :, :, P - 2])
            mu_iL = ((1 / P) * G_iL.sum(dim=1))
            S_iL = torch.cov(G_iL)
            for p in range(P):
                Gij_LR = (x[j, :, p, 0] - x[i, :, p, P-1]).T
                D_LR[i, j] += (Gij_LR - mu_iL) @ torch.inverse(S_iL) @ (Gij_LR - mu_iL).T

            G_jR = x[j, :, :, 0] - x[j, :, :, 1]
            mu_jR = 1 / P * G_jR.sum(dim=1)
            S_jR = torch.cov(G_jR)
            for p in range(P):
                Gij_RL = x[i, :, p, P - 1] - x[j, :, p, 0]
                D_RL[i, j] += (Gij_RL - mu_jR) @ torch.inverse(S_jR) @ (Gij_RL - mu_jR).T

            C_LR[i, j] = D_LR[i, j] + D_RL[i, j]

    W = torch.zeros((n_tiles, n_tiles))
    C_LR.fill_diagonal_(0.)
    for i in range(n_tiles):
        W[i] = C_LR[i] / C_LR[:, i].topk(k=2, largest=False).values[1]

    W = torch.ones((n_tiles, n_tiles)).fill_diagonal_(0.) - W

    C = (1. - torch.eye(*W.shape, device=x.device)) * (W > 0.)
    W = W * C
    W[W > 1 - 1e-4] = 0.
    vals, inds = W.topk(1, dim=1)
    W = torch.zeros_like(W)
    W[torch.arange(W.shape[0], device=W.device).repeat_interleave(1),
      inds.flatten()] = vals.flatten()
    W = torch.ones([W.shape[0], W.shape[1]]) - W
    W[W == 1.] = 0.
    Ch = W

    '''Ch = W
    Ch[Ch < 0.] = 0.'''

    return Ch
