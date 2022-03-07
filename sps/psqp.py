from tqdm import tqdm
import torch
from torch_sparse import coalesce, spmm
from sps.projection_matrix import projection_matrix, lambdas


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

    def sparse_coordinates(tx, ty):  # DA RISCRIVERE IN MODO DA NON USARE PIU' COALESCE
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

    '''
        qui vals is a list of 34 tensors, each tensor is of length ???, with cat, these 34 tensors are combined in one.
        And we get only 1 tensor with len 4896, that are the non-zero values but there are some zeroes.
    '''
    inds, vals = torch.cat(inds, dim=1), torch.cat(vals)
    #inds, vals = coalesce(inds, vals, nt ** 2, nt ** 2)

    ''' funziona tutto anche commentando la riga con coalesce, quindi a che serve?? '''
    '''
        to go from Ch e Cv to A = { inds, vals, ...} happens something i dont know...
    '''

    return torch.sparse_coo_tensor(inds, vals, (nt**2, nt**2)).coalesce()
    # return {"index": inds, "value": vals, "m": nt ** 2, "n": nt ** 2}


'''
 A or vals contains only some value, not all the 144x144 values of the real matrix.
'''
def psqp(A, N, lr=1e-3):
    active = torch.full((N ** 2, 1), fill_value=True, device=A.device)
    p = torch.empty((N ** 2, 1), device=A.device)

    for Na in tqdm(range(1), total=N):
        p[active] = 1. / (N - Na)
        # d = spmm(**A, matrix=p)
        d = torch.sparse.mm(A, p)

        K, Nk = projection_matrix(p, N)
        lambdas_x = lambdas(p, Nk, N)
        s = torch.mm(K.t(), d)

        ''' # tau_m == step, sembra non venire calcolata esattamente bene

         if torch.linalg.norm(step) == 0:
            tau_m =  −(NkTNk)^−1 NkT d

        else: 
            z = s / torch.linalg.norm(s)
            tau_i = torch.div(lambdas_x, (Nk * z.item()).t())
            tau_m = torch.min(tau_i)  # should be a value/coeff

        p[active] += tau_m * s.item()
        '''

        p[active] += lr * s.item()
        # p[active] += lr * d[active]
        p.clamp_(0., 1.)
        active = (p != 0.) | (p != 1.)

    return p.reshape(N, N)


