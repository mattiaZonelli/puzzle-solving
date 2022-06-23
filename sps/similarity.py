import torch


def sim(x1, x2, name):
    if name == "gaussian":
        x1 = x1 / (x1.norm(dim=1, p=float('inf'), keepdim=True) + 1e-12)
        x2 = x2 / (x2.norm(dim=1, p=float('inf'), keepdim=True) + 1e-12)
        sq_norms1 = (x1 ** 2.).sum(dim=1)
        sq_norms2 = (x2 ** 2.).sum(dim=1)
        # squared euclidean distances
        W = sq_norms1.unsqueeze(1) + sq_norms2.unsqueeze(0) - \
            2. * torch.mm(x1, x2.t())

        sigmas = W.topk(3, dim=1, largest=False)[0][:, -1]
        W = (-W / (2 * torch.ger(sigmas, sigmas) + 1e-6)).exp()
    elif name == "correlation":
        x1 = x1 - x1.mean(dim=1, keepdim=True)
        x2 = x2 - x2.mean(dim=1, keepdim=True)
        norms1 = x1.norm(dim=1)
        norms2 = x2.norm(dim=1)
        W = torch.mm(x1, x2.t()) / (torch.ger(norms1, norms2) + 1e-6)

    C = (1. - torch.eye(*W.shape, device=x1.device)) * (W > 0.)
    W = W * C

    k = 4
    vals, inds = W.topk(k, dim=1)
    W = torch.zeros_like(W)
    W[torch.arange(W.shape[0], device=W.device).repeat_interleave(k),
      inds[:, :k].flatten()] = vals[:, :k].flatten()
    return W


def psqp_Ch(t):
    pairs = torch.combinations(torch.arange(t.shape[0]))
    diss = torch.zeros((t.shape[0], t.shape[0]))
    for i, j in pairs:
        T = t.shape[2]
        for k in range(T):
            for l in range(3):
                diss[i][j] += torch.pow(t[i, l, k, T - 1] - t[j, l, k, 0], 2)

    sigmas = torch.zeros(t.shape[0])
    for i in range(t.shape[0]-1):
        sigmas[i] = diss[i, diss[i] > 0].min()
    sigmas = 2 * torch.pow(sigmas, 2)

    W = (- diss / sigmas.unsqueeze(1) + 1e-6).exp()
    W = torch.nan_to_num(W, 0.)
    C = (1. - torch.eye(*W.shape, device=t.device)) * (W > 0.)
    W = W * C
    W[W > 1-1e-4] = 0.
    vals, inds = W.topk(1, dim=1)
    W = torch.zeros_like(W)
    W[torch.arange(W.shape[0], device=W.device).repeat_interleave(1),
      inds.flatten()] = vals.flatten()
    W = torch.ones([W.shape[0], W.shape[1]]) - W
    W[W == 1.] = 0.
    return W


def psqp_Cv(t):
    pairs = torch.combinations(torch.arange(t.shape[0]))
    diss = torch.zeros((t.shape[0], t.shape[0]))
    for i, j in pairs:
        T = t.shape[2]
        for k in range(T):
            for l in range(3):
                diss[i][j] += torch.pow(t[i, l, T - 1, k] - t[j, l, 0, k], 2)

    sigmas = torch.zeros(t.shape[0])
    for i in range(t.shape[0]-1):
        sigmas[i] = diss[i, diss[i] > 0].min()
    sigmas = 2 * torch.pow(sigmas, 2)

    W = (- diss / sigmas.unsqueeze(1) + 1e-6).exp()
    W = torch.nan_to_num(W, 0.)
    C = (1. - torch.eye(*W.shape, device=t.device)) * (W > 0.)
    W = W * C
    W[W > 1-1e-4] = 0.
    vals, inds = W.topk(1, dim=1)
    W = torch.zeros_like(W)
    W[torch.arange(W.shape[0], device=W.device).repeat_interleave(1),
      inds.flatten()] = vals.flatten()
    W = torch.ones([W.shape[0], W.shape[1]]) - W
    W[W == 1.] = 0.
    return W


def myaccuracy(pred, target):
    acc = 0.
    for i in range(target.shape[0]):
        if target[i][target[i].argmax().item()] == pred[i][target[i].argmax().item()]:
            acc += 1/target.shape[0]
    return acc