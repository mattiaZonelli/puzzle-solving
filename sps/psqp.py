import random
from sps.projection_matrix import *
from scipy.optimize import line_search

E = 1e-3


def compatibilities(Ch, Cv, puzzle_size):
    rows, cols = puzzle_size[0].item(), puzzle_size[1].item()
    nt = rows * cols
    blk_comps = (Cv.t(), Ch.t(), Ch, Cv)
    # neigh_inds = [C.nonzero() for C in blk_comps]
    # neigh_vals = [C[C > 0.] for C in blk_comps]
    neigh_vals = [C.flatten() for C in blk_comps]

    def block_range(bx, by):
        return torch.arange(nt) + (bx * cols + by) * nt

    def tiles2blk(t_coo, n_coo):
        blk1_rng, blk2_rng = block_range(*t_coo), block_range(*n_coo)
        return torch.stack(torch.meshgrid(blk1_rng, blk2_rng),
                           dim=2).reshape(-1, 2).t()

    def sparse_coordinates(tx, ty):
        tile_inds, tile_vals = [], []
        neigh_coos = (tx - 1, ty), (tx, ty - 1), (tx, ty + 1), (tx + 1, ty)

        for (nx, ny), n_vals in zip(neigh_coos, neigh_vals):
            if 0 <= nx < rows and 0 <= ny < cols:
                tile_inds.append(tiles2blk((tx, ty), (nx, ny)))
                tile_vals.append(n_vals)
        return tile_inds, tile_vals

    inds, vals = [], []
    for tx in range(rows):
        for ty in range(cols):
            t_inds, t_vals = sparse_coordinates(tx, ty)
            inds += t_inds
            vals += t_vals

    inds, vals = torch.cat(inds, dim=1), torch.cat(vals)
    # inds, vals = coalesce(inds, vals, nt ** 2, nt ** 2)

    return torch.sparse_coo_tensor(inds, vals, (nt ** 2, nt ** 2)).coalesce()
    # return {"index": inds, "value": vals, "m": nt ** 2, "n": nt ** 2}


def psqp_ls(A: torch.tensor, N: int) -> torch.tensor:
    active = torch.full((N ** 2, 1), fill_value=True, device=A.device)
    p = torch.empty((N ** 2, 1), device=A.device)
    pi = torch.full((N,), fill_value=-1, device=A.device)

    Np, Nk, bp, bk = constraints_matrix(N)
    r = random.randrange(0, N * 2)
    Np, Nk, bp, bk = move_constraint(torch.tensor(r), Np, Nk, bp, bk)

    Na = 0

    while Na < N:
        p[active] = 1. / (N - Na)

        d = torch.mm(A, p) * 2.  # gv not null

        Pq = projection_matrix(p.shape[0], Np)
        s = (Pq @ d)
        # s, Np, Nk, bp, bk = adjust_dependency(s, Np, Nk, bp, bk, p.shape[0], d)
        # normalize s with norma max first, then whit norm
        s /= torch.linalg.norm(s, float('inf'))
        s /= torch.linalg.norm(s)

        def obj_foo(x):
            return (x.unsqueeze(0) @ torch.sparse.mm(A, x.unsqueeze(1))).squeeze()

        def obj_grad(x):
            return torch.sparse.mm(A, x.unsqueeze(1)).squeeze()

        step = line_search(obj_foo, obj_grad, p.squeeze(), -s.squeeze())

        while step[0] is not None and torch.max(p[active]) < 1 + E:
            p[active] += step[0] * s[active]
            step = line_search(obj_foo, obj_grad, p.squeeze(), -s.squeeze())

        p = p.reshape(N, N)

        active = active.reshape(N, N)

        if step[0] is None:
            # pi[pi < 0.] = 0.
            # TODO: scegliere tra dim=0 e =1 in base a chi restituisce la Global comp più alta (non so se abbia senso)
            pi = p.argmax(dim=1)
            return p.reshape(N ** 2, 1), pi

        if N - Na == 1:
            pi[torch.where(pi < 0)[0]] = p[torch.where(pi < 0)[0]].argmax()
            Na += 1
        else:
            still_active = torch.nonzero(active)
            for i in range(len(still_active)):
                k, l = still_active[i]
                if p[k][l] <= E:
                    active[k][l] = False
                if p[k][l] > 1 - E:
                    active[k][l] = False
                    # pi[k] = l  # si spacca con puzzle quadrati, side dispari (3x3, 5x5,...)
                    if pi[k] == -1:
                        pi[k] = l
                        Na += 1
        p = p.reshape(N ** 2, 1)
        active = active.reshape(N ** 2, 1)

    pi[pi < 0.] = 0.
    return pi, p


'''
steplength as Rosen
def psqp(A: torch.tensor, N: int) -> torch.tensor:
    active = torch.full((N ** 2, 1), fill_value=True, device=A.device)
    p = torch.empty((N ** 2, 1), device=A.device)
    pi = torch.full((N,), fill_value=0, device=A.device)

    Np, Nk, bp, bk = constraints_matrix(N)
    r = random.randrange(0, N*2)
    Np, Nk, bp, bk = move_constraint(torch.tensor(r), Np, Nk, bp, bk)

    Na = 0

    while Na < N:
        p[active] = 1. / (N - Na)

        d = torch.sparse.mm(A, p) * 2.  # gv not null

        Pq = projection_matrix(p.shape[0], Np)
        s = (Pq @ d)
        # s, Np, Nk, bp, bk = adjust_dependency(s, Np, Nk, bp, bk, p.shape[0], d)

        if -1e-3 <= torch.linalg.norm(s) <= 1e-3:  # S == 0
            # in rosen's paper is called r
            lambdas = ((torch.inverse((Np.T @ Np)) @ Np.T) @ d)
            if torch.max(lambdas) <= 1e-4:  # if ALL lambdas are <= 0
                return pi.int()
            else:  # some components of lambdas are positive
                # drop Hq corresponding to rq = max{ri} > 0
                q = torch.argmax(lambdas).unsqueeze(0)
                Np, Nk, bp, bk = move_constraint(q, Np, Nk, bp, bk)  # drop q-th constraint_ from Np

        else:
            # normalize s with norma max first, then whit norm
            s /= torch.linalg.norm(s, float('inf'))
            s /= torch.linalg.norm(s)
            lambda_M, M = steplength(p, Nk, bk, s, active)

            d_lM = (s.T @ torch.sparse.mm(A, (p + lambda_M * s)))  # z T g'_v+1 (Rosen)
            if d_lM < 0.:  # d_lM < 0
                # interpolation as in Rosen's
                zTg = (s.T @ d)
                rho = zTg / (zTg - d_lM)
                p[active] = (rho * (p[active] + lambda_M * s[active])) + ((1 - rho) * p[active])
            else:  # d_lM >= 0
                if len(M) > 1:
                    M = check_z(s, Nk)
                if M.nelement() != 0:
                    Nk, Np, bk, bp = move_constraint(M, Nk, Np, bk, bp)  # add the q-th constraint to Np

                p[active] += lambda_M * s[active]

            p = p.reshape(N, N)
            # p -= p.min(dim=1).values.unsqueeze(1)
            # assert torch.allclose(torch.sum(p, dim=1), torch.ones(N))
            # assert torch.allclose(torch.sum(p, dim=0), torch.ones(N))
            active = active.reshape(N, N)

            still_active = torch.nonzero(active)
            for i in range(len(still_active)):
                k, l = still_active[i]
                if -E <= p[k][l] <= E:
                    active[k][l] = False
                if p[k][l] > 1. - E:
                    # active[k][l] = False
                    active[k] = torch.zeros((1, N), dtype=bool)
                    active[torch.arange(N), l] = torch.zeros((N,), dtype=bool)
                    pi[k] = l
                    Na += 1
                    p[k] /= p[k].sum()
            for nonact in (active == 0).nonzero():
                p[nonact[0]][nonact[1]] = 1. if p[nonact[0]][nonact[1]] > 1.-E else 0.
            p = p.reshape(N ** 2, 1)
            active = active.reshape(N ** 2, 1)
    return pi.int()
'''

'''     
function compare relaxation and psqp(...)        
        h = 2
        r = 2
        lim_h = 9
        lim_r = 9
        psqp_dacc = torch.zeros(lim_r - r + lim_h - h)
        rl_dacc = torch.zeros(lim_r - r + lim_h - h)
        n_tiles = torch.zeros(lim_r - r + lim_h - h)
        i = 1
        while h < lim_h and r < lim_r:
            order = list(range(h * r))
            random.shuffle(order)
            order = torch.tensor([order]).int()
            # relab = solve_puzzle((h, r), order).int()
            for k in range(20):
                relab = solve_puzzle((h, r), order).int()
                try:
                    t_dacc = self.accuracy(relab.squeeze(), order.squeeze()).item()
                except:
                    t_dacc = my_accuracy(relab.squeeze(), order.squeeze(), h * r)
                if t_dacc > rl_dacc[i]:
                    rl_dacc[i] = t_dacc

            data['puzzle_size'] = torch.tensor([h, r]).unsqueeze(0)
            Ch, Cv = oracle_compatibilities(h, r, order)
            A = compatibilities(Ch, Cv, data["puzzle_size"].squeeze())
            p = psqp_ls(A, N=(h * r))
            try:
                psqp_dacc[i] = self.accuracy(p.squeeze(), order.squeeze()).item()
            except:
                psqp_dacc[i] = my_accuracy(p.squeeze(), order.squeeze(), h * r)
            if psqp_dacc[i] < 1e-3:
                for k in range(15):
                    p = psqp_ls(A, N=(h * r))
                    dacc = my_accuracy(p.squeeze(), order.squeeze(), h * r)
                    if dacc > psqp_dacc[i]:
                        psqp_dacc[i] = dacc

            n_tiles[i] = h * r
            if h == r:
                r += 1
            else:
                h += 1
            i += 1

        fileName = r'n_tile vs accuracy.png'
        fig, ax = plt.subplots(1)
        plt.plot(n_tiles, rl_dacc, label='ReLab Accuracy')
        plt.plot(n_tiles, psqp_dacc, label='PSQP Accuracy')
        plt.legend()
        plt.xlabel('# tiles')
        plt.ylabel('Accuracy')
        plt.show()
        fig.savefig(fileName, format='png')
        plt.close(fig)
'''
