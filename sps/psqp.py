from tqdm import tqdm
import torch
from torch_sparse import coalesce, spmm
from projection_matrix import projection_matrix


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
            inds += t_inds; vals += t_vals
    inds, vals = torch.cat(inds, dim=1), torch.cat(vals)
    inds, vals = coalesce(inds, vals, nt ** 2, nt ** 2)

    return {"index": inds, "value": vals, "m": nt ** 2, "n": nt ** 2}



def psqp(A, N, lr=1e-3):
    active = torch.full((N ** 2, 1), fill_value=True, device=A["value"].device)
    p = torch.empty((N ** 2, 1), device=A["value"].device)
    # K = projection_matrix(p)

    for Na in tqdm(range(1), total=N):
        p[active] = 1. / (N - Na)
        d = spmm(**A, matrix=p)
        p[active] += lr * d[active]
        p.clamp_(0., 1.)
        active = (p != 0.) | (p != 1.)

    return p.reshape(N, N)
