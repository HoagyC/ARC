import collections

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class GridRL(nn.Module):
    def __init__(self, num_list, func_list):
        self.enc = OneHotEncoder(categories=num_list+func_list+[None])
        super(GridRL, self).__init__()
        self.num_list = num_list
        self.func_list = func_list
        self.num_len = len(num_list)
        self.func_len = len(func_list)
        self.out_len = self.num_len + self.func_len
        self.max_len = 40

        self.conv1 = nn.Conv2d(1, 20, 3, 2)
        self.conv2 = nn.Conv2d(20, 20, 3, 2)
        self.fc_prog = nn.Linear(self.max_len * len(self.enc.categories), 50)
        self.fc_im = nn.Linear(720, 50)
        self.fc_both = nn.Linear(100, self.out_len)

    def forward(self, im, prog):
        prog = flatten(prog)
        prog += [None] * (self.max_len - len(prog))
        prog = self.enc.fit(prog)
        print(prog)
        prog = self.fc_prog(prog)

        im = torch.from_numpy(im)
        im = im.float()
        im = im.unsqueeze([0, 1])
        print(im.shape)
        im = self.conv1(im)
        im = f.relu(im)
        print(im.shape)
        im = self.conv2(im)
        im = f.relu(im)
        im = torch.flatten(im, 1)
        im = self.fc_im(im)

        x = torch.cat([prog, im])
        x = self.fc_both(x)

        num_vals = nn.Softmax(x[:self.num_len])
        func_vals = nn.Softmax(x[self.num_len:])

        num = np.random.choice(self.num_list, p=num_vals)
        func = np.random.choice(self.func_list, p=func_vals)

        return [num, func]


def pad_grid(grid: np.ndarray, x: int = 30, y: int = 30):
    grid += 1
    pad_x = x - grid.shape[0]
    pad_y = y - grid.shape[1]
    return np.pad(grid, [[0, pad_x], [0, pad_y]], mode='constant', constant_values=0)


def unpad_grid(grid):
    pass


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


g = GridRL([1], [enumerate])
print(g.enc.categories)