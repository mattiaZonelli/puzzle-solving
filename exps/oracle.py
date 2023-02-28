import torch


def oracle_compatibilities_og(data):
    h, w = data['puzzle_size'][0]
    order = data['order'].reshape(h, w)
    Ch = torch.full((h * w, h * w), 0.)
    Cv = torch.full((h * w, h * w), 0.)
    for i in range(h):
        for j in range(w - 1):
            Ch[order[i][j]][order[i][j+1]] = 1.

    for i in range(h - 1):
        for j in range(w):
            Cv[order[i][j]][order[i+1][j]] = 1.
    return Ch, Cv


def oracle_compatibilities(h, w, order):
    order = order.reshape(h, w)
    Ch = torch.full((h * w, h * w), 0.)
    Cv = torch.full((h * w, h * w), 0.)
    for i in range(h):
        for j in range(w - 1):
            # Ch[i * w + j][i * w + j + 1] = 1.
            Ch[order[i][j]][order[i][j + 1]] = 1.
            # Ch[i * w + j + 1][i * w + j] = 1.

    for i in range(h - 1):
        for j in range(w):
            #Cv[i * w + j][(i + 1) * w + j] = 1.
            Cv[order[i][j]][order[i + 1][j]] = 1.
            # Cv[(i + 1) * w + j][i * w + j] = 1.
    return Ch, Cv


# def puzzle_accuracy(target, actual, n, c_tiles):
def puzzle_accuracy(target, actual, n):
    acc = 0.
    for i in range(n):
        if target[i] == actual[i]:
            acc += 1/n
        # elif target[i] in c_tiles:
        #     acc += 1 / n
    return acc