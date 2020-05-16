import json
import collections
import random
import copy

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.nn.functional as f

from matplotlib import pyplot as plt

from os import listdir
import time

from DSL_functions import get_func_list, get_num_list, evaluate_flat


class GridRL(nn.Module):
    def __init__(self, num_list, func_list):
        super(GridRL, self).__init__()
        self.num_list = list(dict.fromkeys(num_list))
        self.func_list = list(dict.fromkeys(func_list))

        enc_cats = self.num_list + self.func_list + ['none']
        enc_cats = np.array([str(i) for i in enc_cats])

        self.enc = OneHotEncoder(categories=[np.unique(enc_cats)])
        self.num_len = len(self.num_list)
        self.func_len = len(self.func_list)
        self.out_len = self.num_len + self.func_len
        self.max_len = 40

        self.conv1 = nn.Conv2d(2, 20, 3, 2)
        self.conv2 = nn.Conv2d(20, 20, 3, 2)
        self.fc_prog = nn.Linear(self.max_len * len(self.enc.categories[0]), 50).double()
        self.fc_im = nn.Linear(720, 50)
        self.fc_both = nn.Linear(100, self.out_len).double()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, im, prog):

        prog = prog.double()
        prog = torch.flatten(prog, 1)
        im = im.float()

        prog = self.fc_prog(prog)
        im = self.conv1(im)
        im = f.relu(im)
        im = self.conv2(im)
        im = f.relu(im)
        im = torch.flatten(im, 1)
        im = self.fc_im(im).double()

        x = torch.cat([prog, im], dim=1)
        x = self.fc_both(x)

        return x


def plot_loss(losses, smoothing=500):
    smoothed_losses = [np.mean(losses[i:i+smoothing]) for i in range(len(losses) - smoothing)]
    plt.plot(smoothed_losses)
    plt.show()


def train_model(model_path: str = None, batch_size=50, epoch_n=5, n_prog=200):
    losses = []
    func_list = get_func_list()
    num_list = get_num_list()
    model = GridRL(num_list, func_list)

    if model_path:
        model.load_state_dict(torch.load(model_path))
        model.eval()

    data_t = 0
    model_t = 0
    for epoch in range(epoch_n):
        programs = [build_flat_random(func_list, num_list, 1, []) for _ in range(n_prog)]
        for _ in range(20):
            t = time.time()
            data = make_training_data(programs, model.enc, model.max_len, 10000)
            data_t += time.time() - t
            t = time.time()
            random.shuffle(data)
            print('epoch examples:', len(data))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            n_batches = len(data) // batch_size
            max_loss = 0
            for i in range(n_batches):
                batch = data[i * batch_size:(i + 1) * batch_size]
                im_in = torch.FloatTensor([a[0] for a in batch])
                im_out = np.array([a[1] for a in batch])
                in_prog = np.array([a[2] for a in batch])
                target = np.array([a[3] for a in batch])

                im_comb = np.stack([im_in, im_out], 1)
                im_comb = torch.from_numpy(im_comb)
                in_prog = torch.from_numpy(in_prog)
                guess = model(im_comb, in_prog)
                target = torch.Tensor(target)
                target = target.type(torch.LongTensor)

                # print(target, guess)
                loss = model.cel(guess, target)
                # print(loss.data, guess.shape, target.shape, guess[0], target[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.data)

        model_t += time.time() - t
        print(f'Epoch {epoch} complete, average loss: {np.mean(losses[-n_batches * 20:])}')
        print('data %', round(data_t/model_t), 'total_time', round(data_t + model_t))

    plot_loss(losses)

    if not model_path:
        current_time = time.strftime("%Y%m%d.%H%M%S", time.localtime())
        model_path = "models/" + current_time
    torch.save(model.state_dict(), model_path)


def make_training_data(programs, enc, max_len, data_len):
    init_images = get_input_images()
    data = []
    while len(data) < data_len:
        for i in init_images:
            cp = random.choice(programs)

            im = evaluate_flat(cp, i)
            im = pad_grid(im)
            i = pad_grid(i)
            cp = list(flatten(cp))
            cp = [str(i) for i in cp]
            for cut in range(len(cp)):
                target = cp[cut]
                cut_p = cp[:cut]
                cut_p += ['none'] * (max_len - len(cut_p))
                cut_p = np.asarray(cut_p)
                cut_p = np.reshape(cut_p, (-1, 1))
                p_data = enc.fit_transform(cut_p).toarray()
                l = list(enc.categories[0])
                target = l.index(target)
                data.append((i, im, p_data, target))
    return data


def get_input_images():
    data_dir = "./data/evaluation"
    training_files = sorted(listdir(data_dir))

    tasks = []
    for task_file in training_files:
        with open("/".join([data_dir, task_file])) as f:
            task_id = task_file.split(".")[0]
            tasks.append((task_id, json.load(f)))

    init_images = []
    for t in tasks:
        init_images += [np.array(x['input']) for x in t[1]['train']]

    return init_images


def pad_grid(grid: np.ndarray, x: int = 30, y: int = 30):
    if grid.shape[0] > x:
        grid = grid[:x, :]
    if grid.shape[1] > y:
        grid = grid[:, :x]
    if grid.dtype == bool:
        grid = grid.astype(np.int64)
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


def build_flat_random(func_list, num_list, prog_len=1, start=[]):
    prog = start

    def add_function(prog):
        add = random.choice(func_list)
        prog += [add]
        add_args(prog, add)
        return

    def add_args(prog, func):
        if func.numbers:
            nums = random.choices(num_list, k=func.numbers)
            for n in nums:
                prog += [n]
                if type(n) != int:
                    add_args(prog, n)

        if func.grids:
            funcs = random.choices(func_list, k=func.grids)
            for n in funcs:
                prog += [n]
                add_args(prog, n)

        return

    for _ in range(prog_len):
        add_function(prog)

    return prog


def build_flat_model(model, t_input, t_output, prog_len=1, start=[], temp=10):

    func_list = model.func_list
    num_list = model.num_list

    def prep_prog(prog):
        p = prog.copy()
        p = [str(i) for i in p]
        p += ['none'] * (model.max_len - len(p))
        p = np.asarray(p)
        p = np.reshape(p, (-1, 1))
        p = model.enc.fit_transform(p).toarray()
        p = np.expand_dims(p, 0)
        p = torch.from_numpy(p)
        return p

    def get_new(prog, add_type, temp):
        if len(prog) >= model.max_len - 1:
            raise ValueError
        input_prog = prep_prog(prog)
        im_comb = np.stack([t_input, t_output], 0)
        im_comb = np.expand_dims(im_comb, 0)
        im_comb = torch.from_numpy(im_comb)

        if add_type == 'func':
            with torch.no_grad():
                choice = model(im_comb, input_prog).numpy()[0]
                weights = softmax(choice[len(num_list):] / temp)
            return np.random.choice(func_list, p=weights)

        elif add_type == 'num':
            with torch.no_grad():
                choice = model(im_comb, input_prog).numpy()[0]
                weights = softmax(choice[:len(num_list)] / temp)
            return np.random.choice(num_list, p=weights)

    def add_tree(prog, add_type, temp):

        new = get_new(prog, add_type, temp)

        prog += [new]

        if type(new) == int:
            return prog

        if new.numbers:
            for _ in range(new.numbers):
                prog = add_tree(prog, 'num', temp)

        if new.grids:
            for _ in range(new.grids):
                prog = add_tree(prog, 'func', temp)

        return prog

    prog = copy.copy(start)
    restart = True
    while restart:
        restart = False
        try:
            prog = copy.copy(start)
            for _ in range(prog_len):
                    prog = add_tree(prog, 'func', temp)
        except ValueError:
            restart = True

    return prog


if __name__ == "__main__":
    train_model(model_path='models/20200508.201046')
