# TO DO:

# Need functions:
# Shift, maybe especially colour specific
# Get
# Match to points

# Should add cache to especially rectangulate and enclosed

import numpy as np
from operator import itemgetter
import copy
import collections

import time

from visualising import show_graph

def attrs(**attrs):
    def with_attrs(f):
        for k,v in attrs.items():
            setattr(f, k, v)
        return f
    return with_attrs


@attrs(numbers=2, points=0, grids=0)
def rectangulate(x, n, m):
    recs = []
    bool_grid = x.copy()
    t = time.time()
    # Rectangles given in form (x, y, x_len, y_len)
    while bool_grid.any():
        if len(recs) > 9:
            break
        if time.time() - t > 10:
            print('rectangulate failed')
            print(x)
            break
        rows = 1
        start = end = 0
        for i, r in enumerate(bool_grid):
            if not any(r) and end == 0:
                continue
            elif end:
                if all(r[start:end]):
                    rows += 1
                else:
                    break
            else:
                start = np.argmax(r)
                if start == len(r) - 1:
                    end = start + 1
                else:
                    if all(r[start:]):
                        end = len(r)
                    else:
                        end = np.argmin(r[start:]) + start
                start_row = i

        recs.append([start_row, start, rows, end - start])
        bool_grid[start_row : start_row + rows, start:end] = False

    recs.sort(key=lambda x: -x[2] * x[3])
    x.fill(0)
    if n < len(recs):
        rec = recs[n]
        x[rec[0]:rec[0]+rec[2], rec[1]:rec[1] + rec[3]] = m
    x.reshape(x.shape[0], -1)
    return x


@attrs(numbers=2, points=0, grids=0)
def rectangulate_(x, n, m):
    recs = []
    y = x.copy()
    xl, yl = x.shape
    for i in range(xl):
        for j in range(yl):
            if i > 0 and y[i, j] == y[i-1, j]:
                continue
            if j > 0 and y[i, j] == y[i, j-1]:
                continue
            l, h = 1, 1
            while l + i <= xl:
                if all(y[i, j] == y[i:i+1+1, j]):
                    l = l + 1
                else:
                    break

            while h + j <= yl:
                if all(y[i, j] == y[i:i+l+1, j:j+h+1]):
                    h = h + 1
                else:
                    break

            recs.append(i, j, l, h)

    recs.sort(key=lambda x: -x[2] * x[3])

    y.fill(0)
    if n < len(recs):
        rec = recs[n]
        y[rec[0]:rec[0]+rec[2], rec[1]:rec[1] + rec[3]] = m
    y.reshape(y.shape[0], -1)
    return y


@attrs(numbers=2, points=0, grids=0)
def wrap(x, n, m):
    try:
        r = np.pad(x, ((0, n), (0, m)), 'wrap')
    except:
        r = x
    return r


@attrs(numbers=2, points=0, grids=0)
def swap_color(x, n, m):
    c = x.copy()
    c[x == n] = m
    return c


def get_period_length(x):
    h, w = x.shape
    period = 1
    while period < h:
        cycled = np.pad(x[:period, :], ((0, h - period), (0, 0)), 'wrap')
        if (cycled == x).all():
            return period
        period += 1
    return 1


@attrs(numbers=1, points=0, grids=0)
def clip(x, n):
    x = x[:n, :]
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    return x


@attrs(numbers=0, points=0, grids=1)
def logical_and(x, y):
    shape = np.minimum(x.shape, y.shape)
    y = y[:shape[0], :shape[1]]
    x = x[:shape[0], :shape[1]]
    return x & y


@attrs(numbers=2, points=0, grids=0)
def upsample(x, n, m):
    n = min(5, n)
    m = min(5, m)
    return x.repeat(n, axis=0).repeat(m, axis=1)


@attrs(numbers=2, points=0, grids=0)
def tile(x, n, m):
    n = min(5, n)
    m = min(5, m)
    return np.tile(x, (n, m))


@attrs(numbers=1, points=0, grids=1)
def color_map(x, col, y):
    shape = np.minimum(x.shape, y.shape)
    y = y[:shape[0], :shape[1]]
    x = x[:shape[0], :shape[1]]
    x[y > 0] = col
    return x


@attrs(numbers=1, points=0, grids=0)
def select_col(x, col):
    return x == col


@attrs(numbers=0, points=0, grids=0)
def crop(x):
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(x)
    if len(true_points) == 0:
        return x
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    x = x[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    return x


@attrs(numbers=0, points=0, grids=0)
def enclosed(x):
    # depth first search
    H, W = x.shape
    Dy = [0, -1, 0, 1]
    Dx = [1, 0, -1, 0]
    arr_padded = np.pad(x, ((1, 1), (1, 1)), "constant", constant_values=0)
    searched = np.zeros(arr_padded.shape, dtype=bool)
    searched[0, 0] = True
    q = [(0, 0)]
    while q:
        y, x = q.pop()
        for dy, dx in zip(Dy, Dx):
            y_, x_ = y + dy, x + dx
            if not 0 <= y_ < H + 2 or not 0 <= x_ < W + 2:
                continue
            if not searched[y_][x_] and arr_padded[y_][x_] == 0:
                q.append((y_, x_))
                searched[y_, x_] = True
    res = searched[1:-1, 1:-1]
    return ~res


@attrs(numbers=0, points=0, grids=1)
def pipe(x, y):
    shape = np.minimum(x.shape, y.shape)
    y = y[:shape[0], :shape[1]]
    x = x[:shape[0], :shape[1]]
    return x | y


@attrs(numbers=0, points=0, grids=1)
def top_colours(x, n):
    freq = sorted([(x == i).sum() for i in range(10)])
    return freq


@attrs(numbers=0, points=0, grids=0)
def identity(x: [np.array]):
    return x


@attrs(numbers=0, points=0, grids=0)
def not_(x: [np.array]):
    return ~x


@attrs(numbers=0, points=0, grids=0)
def flip(x):
    return np.flip(x)


@attrs(numbers=1, points=0, grids=0)
def rotate(x, n):
    return np.rot90(x, n)


@attrs(numbers=0, points=0, grids=1)
def count(x, y):
    return min(9, np.argwhere(y).sum())


@attrs(numbers=0, points=0, grids=1)
def concatv(x, y):
    size = np.minimum(x.shape[1], y.shape[1])
    y = y[:, :size]
    x = x[:, :size]
    return np.concatenate((x, y), 0)


@attrs(numbers=0, points=0, grids=1)
def concath(x, y):
    size = np.minimum(x.shape[0], y.shape[0])
    y = y[:size, :]
    x = x[:size, :]
    return np.concatenate((x, y), 1)


@attrs(numbers=1, points=0, grids=1)
def count_col(x, y, n):
    return min(9, np.argwhere(y == n).sum())


@attrs(numbers=1, points=0, grids=0)
def top_col(x, n):
    freq = [(x == i).sum() for i in range(10)]
    sort_freq = sorted(enumerate(freq), key=itemgetter(1))
    return sort_freq[-n][0]


@attrs(numbers=2, points=0, grids=0)
def roll(x, n, m):
    return np.roll(x, (n, m))


# @attrs(numbers=2, points=0, grids=0)
# def roll_col(x, n, m):
#     x = x.fill(0)
#     x(np.argwhere(x == col)
#     x & y
#     return np.roll(x, (n, m))


def get_func_list():
    return [identity, identity, identity, swap_color, tile, upsample, color_map, enclosed, clip, logical_and, wrap,
            select_col, flip, rotate, crop, rectangulate, not_, concath, concatv]


def get_num_list():
    return [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, count_col, count, top_col]


# Takes a program in the DSL and applies it to a grid.
def evaluate_flat(program: [], input_image: np.array):
    # Make sure the input is a np.array
    in_image = copy.copy(input_image)
    image = copy.copy(input_image)
    assert type(input_image) == np.ndarray
    c = 0

    def sub_eval(image, c, prog):
        current = prog[c]
        args = []
        c += 1

        if type(current) == int:
            return current, c

        if current.numbers:
            for _ in range(current.numbers):
                new_arg, c = sub_eval(in_image, c, prog)
                args += [new_arg]

        if current.grids:
            for _ in range(current.grids):
                new_arg, c = sub_eval(in_image, c, prog)
                args += [new_arg]
        try:
            return current(image, *args), c
        except AttributeError:
            print(prog, current)
            show_graph(prog)

    while c < len(program):
        image, c = sub_eval(image, c, program)

    return image


# Takes a program in the DSL and applies it to a grid.
def evaluate(program: [], input_image: np.array):
    # Make sure the input is a np.array
    image = copy.copy(input_image)
    assert type(input_image) == np.ndarray

    for i, x in enumerate(program):
        if not callable(x):
            continue

        args = program[i+1]
        final_args = []
        numbers = args[0]
        points = args[1]
        grids = args[2]
        for n in numbers:
            if type(n) == list:
                final_args.append(min(9, evaluate(n, input_image)))
            else:
                assert type(n) == int
                final_args.append(n)

        for p in points:
            if type(p) == list:
                final_args.append(evaluate(p, input_image))
            else:
                assert type(p) == tuple
                final_args.append(p)

        for g in grids:
            if type(g) == list:
                result = evaluate(g, input_image)
                shape_crop = np.minimum(result.shape, (25, 25))
                result = result[:shape_crop[0], :shape_crop[1]]
                final_args.append(result)
        image = x(image, *final_args)

    return image

