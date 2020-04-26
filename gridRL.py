# generate_examples(func_list, n_examples)
# plonk in a mini trans

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class GridRL(nn.Module):
    def __init__(self, num_list, func_list):
        self.enc = OneHotEncoder(categories=num_list+func_list)
        super(GridRL, self).__init__()
        self.num_len = len(num_list)
        self.func_len = len(func_list)
        self.out_len = self.num_len + self.func_len
        self.conv1 = nn.Conv2d(1, 20, 3, 2)
        self.conv2 = nn.Conv2d(20, 20, 3, 2)
        self.fc1 = nn.Linear(720, self.out_len)

    def forward(self, im, prog):
        prog = self.enc.fit(prog)
        im = torch.from_numpy(im)
        im = im.float()
        im = im.unsqueeze([0, 1])
        print(im.shape)
        im = self.conv1(im)
        im = f.relu(im)
        print(im.shape)
        im = self.conv2(im)
        im = f.relu(im)

        x = torch.flatten(im, 1)
        x = self.fc1(x)
        return x


def pad_grid(grid: np.ndarray, x: int = 30, y: int = 30):
    grid += 1
    pad_x = x - grid.shape[0]
    pad_y = y - grid.shape[1]
    return np.pad(grid, [[0, pad_x], [0, pad_y]], mode='constant', constant_values=0)


def unpad_grid(grid):
    pass
