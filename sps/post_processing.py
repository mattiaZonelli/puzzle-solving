import copy
import random
import torch
from scipy.stats import ks_2samp
from torchmetrics import Accuracy

E = 1e-3

# aggiungo i pezzi che non erano stati inseriti
def missing_tiles2(pi, p, ACC, const_tiles, tiles_order):
    n_tiles = pi.shape[0]
    accuracy = Accuracy(num_classes=n_tiles)
    uniques, counts = pi.unique(return_counts=True)

    missing_t = torch.arange(n_tiles)
    for i in range(n_tiles):
        if i in uniques or i in const_tiles:
            missing_t[i] = -1
    missing_t = missing_t[missing_t > -1]

    for mt in missing_t:
        tmp = copy.deepcopy(pi)
        for i in range(n_tiles):
            if mt not in pi:
                tmp[i] = mt
                dacc = accuracy(tmp, tiles_order).item()
                if dacc >= ACC and len(tmp.unique()) > len(pi.unique()):
                    pi = copy.deepcopy(tmp)
                    ACC = dacc
                else:
                    tmp = copy.deepcopy(pi)

    for i in range(n_tiles):
        tmp = torch.zeros_like(pi)
        tmp[pi[i]] = 1.
        p.reshape(n_tiles, n_tiles)[i] = copy.deepcopy(tmp)

    return pi, p

# inserisco anche le constant tiles
def missing_tiles3(pi, p, ACC, const_tiles, tiles_order):
    n_tiles = pi.shape[0]
    accuracy = Accuracy(num_classes=n_tiles)
    uniques, counts = pi.unique(return_counts=True)
    missing_t = torch.cat((const_tiles.squeeze(), uniques[counts > 1]), 0)

    for i in range(n_tiles):
        if pi[i] in missing_t:
            pi[i] = -1

    tmp_pi = copy.deepcopy(pi)
    tmp_p = copy.deepcopy(p)
    final_pi = copy.deepcopy(pi)
    final_p = copy.deepcopy(p)
    for t in range(20):
        order = list(range(missing_t.shape[0]))
        random.shuffle(order)
        missing_t = missing_t[order]
        k = 0
        for m in torch.where(pi < 0.)[0]:
            tmp = torch.zeros_like(pi)
            tmp[missing_t[k]] = 1.
            p.reshape(n_tiles, n_tiles)[m] = copy.deepcopy(tmp)
            pi[m] = missing_t[k]
            k += 1

        dacc = accuracy(pi, tiles_order)
        if dacc >= ACC:
            ACC = dacc
            final_pi = copy.deepcopy(pi)
            final_p = copy.deepcopy(p)

        pi = copy.deepcopy(tmp_pi)
        p = copy.deepcopy(tmp_p)

    return final_pi, final_p




def missing_tiles(p, A, pi, tile_order):
    # global_comp = p.T @ torch.mm(A, p)
    final_result = torch.zeros(pi.shape[0])
    uniques, counts = pi.unique(return_counts=True)
    # tengo una sola occorenza per ogni tile
    # prendo gli indici da pi di dove è il numero, guardare il valore massimo di p in quelle posizione e tenere solo il max
    p_max = p.max(dim=1).values
    tmp = torch.full((pi.shape[0],), fill_value=-1)
    for i in range(pi.shape[0]):
        if pi[i] == tile_order[i]:
            tmp[i] = pi[i]
    for n in uniques:
        if n not in tmp:
            tmp[torch.where(p_max == p_max[torch.where(pi == n)[0]].max())] = n

    # sistemo p in modo che torni ad essere stoccastica
    p = torch.full((pi.shape[0], pi.shape[0]), fill_value=0.)
    for i in torch.where(tmp > -1)[0]:
        p[i, pi[i]] = 1.

    # mi trovo i pezzi che mi mancano
    combined = torch.cat((pi.unique(), torch.arange(pi.shape[0])))
    uniques, counts = combined.unique(return_counts=True)
    missing_tiles = uniques[counts == 1]

    # cerco la permutazione delle missing tiles che ha una Gcomp maggiore
    SIDE = pi.shape[0]

    initial_Gcomp = p.view(SIDE * SIDE).unsqueeze(0) @ torch.mm(A, p.view(SIDE * SIDE).unsqueeze(1))

    # prima di iniziare a lavorare sulle tiles mi salvo in pi dove mancavano pezzi
    pi = copy.deepcopy(tmp)
    final_p = copy.deepcopy(p)
    init_p = copy.deepcopy(p)
    for i in range(20):
        # trova una permutazione di missing_tiles
        order = list(range(missing_tiles.shape[0]))
        random.shuffle(order)
        missing_tiles = missing_tiles[order]
        # inseriscile in tmp
        k = 0
        for m in torch.where(tmp < 0)[0]:
            # aggiungi la tile
            tmp[m] = missing_tiles[k]
            # aggiorna p
            p[m, missing_tiles[k]] = 1.
            k += 1

        # check is Gcomp is greater than before, if yes save the permutation and update Gcomp
        tmp_Gcomp = p.view(SIDE * SIDE).unsqueeze(0) @ torch.mm(A, p.view(SIDE * SIDE).unsqueeze(1))
        if tmp_Gcomp > initial_Gcomp:
            initial_Gcomp = tmp_Gcomp
            final_result = copy.deepcopy(tmp)
            final_p = copy.deepcopy(p)
        # resetto tmp e p alla versione coi "buchi"
        tmp = copy.deepcopy(pi)
        p = copy.deepcopy(init_p)

    '''fine posizionamento missing tiles'''
    #print(final_result)
    #print(final_p)
    p = copy.deepcopy(final_p).reshape(SIDE * SIDE, 1)

    return final_result, p


def cyclical_shift(A, p, pi, dims):
    r, c = dims
    SIDE = r * c
    # cyclical shift on p
    for t in range(20):
        # shift per riga delle colonna
        t_Gcomp = torch.zeros(r * c)
        for i in range(r):
            for j in range(c):
                t_Gcomp[(i * c) + j] = p.T @ torch.mm(A, p)
                p.reshape(r, c, SIDE)[i, :, :] = p.reshape(r, c, SIDE)[i, :, :].roll(1, 0)
                pi = p.reshape(SIDE, SIDE).argmax(dim=1).reshape(r, c)

        # prendo lo shift che ha generato una Gcomp maggiore
        for i in range(r):
            shift = int(t_Gcomp.reshape(r, c).argmax(dim=1)[i])
            p.reshape(r, c, SIDE)[i, :, :] = p.reshape(r, c, SIDE)[i, :, :].roll(shift, 0)
            pi = p.reshape(SIDE, SIDE).argmax(dim=1).reshape(r, c)

        # shift per colonna delle righe
        t_Gcomp = torch.zeros(r * c)
        for i in range(c):
            for j in range(r):
                t_Gcomp[(i * r) + j] = p.T @ torch.mm(A, p)
                p.reshape(r, c, SIDE)[:, i, :] = p.reshape(r, c, SIDE)[:, i, :].roll(1, 0)
                pi = p.reshape(SIDE, SIDE).argmax(dim=1).reshape(r, c)
        for i in range(c):
            shift = int(t_Gcomp.reshape(c, r).argmax(dim=1)[i])
            p.reshape(r, c, SIDE)[:, i, :] = p.reshape(r, c, SIDE)[:, i, :].roll(shift, 0)
            pi = p.reshape(SIDE, SIDE).argmax(dim=1).reshape(r, c)

    return p.argmax(dim=1), p


def constant_tiles(tiles):
    '''
        per ogni tiles, mi creo un tensore con le sue possibili rotazioni. Mi prendo sempre le prime tre righe di pixel,
        che sarebbero i 4 bordi. Calcolo degli istogrammi per ognuno e uso il Kosmogorov-Smirnov test tra tutte le combinazioni.
        in base al p-value, decido se è una constant tiles o meno
    '''
    n_tiles = tiles.shape[0]
    const_tiles = torch.zeros(n_tiles)
    # tensor with the 4 possible rotation
    tmp = torch.empty((4, tiles.shape[1], tiles.shape[2], tiles.shape[3]))
    for i in range(n_tiles):
        tmp[0] = tiles[i]
        tmp[1] = torch.rot90(tiles[i], -1, [1, 2])
        tmp[2] = torch.rot90(tiles[i], -2, [1, 2])
        tmp[3] = torch.rot90(tiles[i], -3, [1, 2])

        borders = tmp[:, :, 0:3, :]
        color_hists = torch.empty((4, 100))
        for j in range(4):
            color_hists[j] = torch.histc(borders[j])
        # gli istogrammi andrebbero normalizzati?
        pairs = torch.combinations(torch.arange(4))
        decision_mat = torch.zeros((4, 4))
        for a, b in pairs:
            decision_mat[a, b] = ks_2samp(color_hists[a], color_hists[b]).pvalue
        if decision_mat[decision_mat > 0.].mean() > 0.05:
            const_tiles[i] = decision_mat[decision_mat > 0.].mean()

    const_tiles = torch.where(const_tiles > const_tiles[const_tiles > 0.].mean(), 1., 0.)

    return const_tiles