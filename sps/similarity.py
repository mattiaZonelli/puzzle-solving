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
        W = (-W / (torch.ger(sigmas, sigmas) + 1e-6)).exp()
    elif name == "correlation":
        x1 = x1 - x1.mean(dim=1, keepdim=True)
        x1 = x2 - x2.mean(dim=1, keepdim=True)
        norms1 = x1.norm(dim=1)
        norms2 = x2.norm(dim=1)
        W = torch.mm(x1, x2.t()) / (torch.ger(norms1, norms2) + 1e-6)

    C = (1. - torch.eye(*W.shape, device=x1.device)) * (W > 0.)
    W = W * C

    vals, inds = W.topk(3, dim=1)
    W = torch.zeros_like(W)
    W[torch.arange(W.shape[0], device=W.device).repeat_interleave(2),
      inds[:, 1:].flatten()] = vals[:, 1:].flatten()
    return W
