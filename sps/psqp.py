import torch
from torch_sparse import coalesce, spmm


def compatibilities(Ch, Cv, puzzle_size):
    rows, cols = puzzle_size
    nt = Ch.shape[0]

    # TODO: replace torch.arange(nt) with these to improve sparseness of matrix
    # Ch_inds, Cv_inds = Ch.nonzero(), Cv.nonzero()
    # Ch_vals, Cv_vals = Ch[Ch > 0.], Cv[Cv > 0.]

    def block_range(bx, by):
        return torch.arange(nt) + (bx * cols + by) * nt

    def tiles2blk(t_coo, n_coo):
        blk1_rng, blk2_rng = block_range(*t_coo), block_range(*n_coo)
        return torch.stack(
            torch.meshgrid(blk1_rng, blk2_rng), dim=2).reshape(-1, 2).t()

    def sparse_coordinates(tx, ty):
        tile_inds, tile_vals = [], []
        neigh_coos = (tx - 1, ty), (tx - 1, ty), (tx + 1, ty), (tx, ty + 1)
        neigh_vals = (Cv.t(), Ch.t(), Ch, Cv.t())

        for (nx, ny), neigh_vals in zip(neigh_coos, neigh_vals):
            if 0 <= nx < rows and 0 <= nx < cols:
                tile_inds.append(tiles2blk((tx, ty), (nx, ny)))
                tile_vals.append(neigh_vals) 
        return tile_inds, tile_vals

    inds, vals = [], []
    for tx in range(rows):
        for ty in range(cols):
            t_inds, t_vals = sparse_coordinates(tx, ty)
            inds.append(t_inds); vals.append(t_vals)
    inds, vals = coalesce(inds, vals, rows, cols)
    
    return {"index": inds, "value": vals, "m": rows, "n": cols}


def psqp(Ch, Cv, N, lr=1e-3):
    A = compatibilities(Ch, Cv)
    active = torch.full((N, N), fill_value=True, device=Ch.device)
    p = torch.empty((N, N), device=Ch.device)

    for Na in range(N):
        p[active] = 1. / (N - Na)
        d = spmm(**A, p.flatten())
        p[active] += lr * d
        p.clamp_(0., 1.)
        active = (p != 0.) | (p != 1.)

    return p
