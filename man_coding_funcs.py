import numpy as np
import random
from os import listdir
import json
import copy

from visualising import plot_grids

def attrs(**attrs):
    def with_attrs(f):
        for k,v in attrs.items():
            setattr(f, k, v)
        return f
    return with_attrs


@attrs(numbers=2, points=0, grids=0)
def wrap(x, n, m):
    return np.pad(x, ((0, n), (0, m)), 'wrap')


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
    return x[:n, :]


@attrs(numbers=0, points=0, grids=1)
def logical_and(x, y):
    shape = np.minimum(x.shape, y.shape)
    y = y[:shape[0], :shape[1]]
    x = x[:shape[0], :shape[1]]
    return x & y


@attrs(numbers=2, points=0, grids=0)
def upsample(x, n, m):
    return x.repeat(n, axis=0).repeat(m, axis=1)


@attrs(numbers=2, points=0, grids=0)
def tile(x, n, m):
    return np.tile(x, (n, m))


@attrs(numbers=1, points=0, grids=1)
def color_map(x, col, y):
    shape = np.minimum(x.shape, y.shape)
    print(shape)
    y = y[:shape[0], :shape[1]]
    x = x[:shape[0], :shape[1]]
    x[y > 0] = col
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
    return res


@attrs(numbers=0, points=0, grids=1)
def pipe(x, y):
    shape = np.minimum(x.shape, y.shape)
    y = y[:shape[0], :shape[1]]
    x = x[:shape[0], :shape[1]]
    return x | y


@attrs(numbers=0, points=0, grids=0)
def identity(x: [np.array]):
    return x


identity.numbers = 0
identity.grids = 0
identity.points = 0


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
        print(numbers)
        for n in numbers:
            if type(n) == list:
                final_args.append(evaluate(n, input_image))
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
                final_args.append(evaluate(g, input_image))

        print("args", final_args)
        image = x(image, *final_args)

    return image


def new_build(len=1):
    functions = [identity, swap_color, tile, upsample, color_map, enclosed, clip, logical_and, wrap]
    points = [(0, 0)]
    numbers = [1, 2, 3, 4, 5]

    program = []

    def add_function():
        new_fn, add_nums, add_points, add_grids = [], [], [], []

        add = random.choice(functions)
        if add.numbers:
            add_nums = random.choices(numbers, k=add.numbers)

        if add.points:
            add_points = random.choices(points, k=add.points)

        if add.grids:
            add_grids = add_function()

        new_fn = [add, [add_nums, add_points, add_grids]]

        return new_fn

    for _ in range(len):
        program += add_function()

    return program



if __name__ == "__main__":
    training_dir = "./data/training"
    training_files = sorted(listdir(training_dir))

    tasks = []
    for task_file in training_files:
        with open("/".join([training_dir, task_file])) as f:
            task_id = task_file.split(".")[0]
            tasks.append((task_id, json.load(f)))

    image = np.array(tasks[121][1]['train'][0]['input'])
    while True:
        program = new_build(len=5)
        print(program)
        new_image = evaluate(program, image)
        print(new_image)
        plot_grids([image, new_image])

# train000 = [upsample_c(3, 3), tile_c(3, 3), logical_and_c]
# train001 = [enclosed, pipe_c(), color_map_c()]
# train002 = [get_period_length, clip_c(), wrap_c(), change_color_c()]


