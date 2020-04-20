import numpy as np
import random
from os import listdir
import os
import json
import time
import copy
from operator import itemgetter

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
    return x.repeat(n, axis=0).repeat(m, axis=1)


@attrs(numbers=2, points=0, grids=0)
def tile(x, n, m):
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


@attrs(numbers=0, points=0, grids=0)
def flip(x):
    return np.flip(x)


@attrs(numbers=1, points=0, grids=0)
def rotate(x, n):
    return np.rot90(x, n)


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

        image = x(image, *final_args)

    return image


def build(prog_len=1, start=[]):
    functions = [identity, swap_color, tile, upsample, color_map, enclosed, clip, logical_and, wrap, select_col,
                 flip, rotate, crop]
    points = [(0, 0)]
    numbers = [1, 2, 3, 4, 5]

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

    for _ in range(prog_len):
        start += add_function()

    return start


def total_fitness(p, e):
    if len(p.shape) == 1:
        p = np.expand_dims(p, axis=0)
    shape = (min(p.shape[0], e.shape[0]), min(p.shape[1], e.shape[1]))
    incorrect = (p[0:shape[0], 0:shape[1]] != e[0:shape[0], 0:shape[1]]).sum()
    misshape = (abs(p.shape[0]-e.shape[0]) * e.shape[1]) + (abs(p.shape[1] - e.shape[1]) * e.shape[0])
    # print(p.shape, e.shape, shape, incorrect, misshape)
    return incorrect + misshape


def evaluate_fitness(program, task):
    """ Take a program and a task, and return its fitness score as a tuple. """
    score = 0

    # For each sample
    for sample in task:
        i = np.array(sample['input'])
        o = np.array(sample['output'])

        # For each fitness function
        images = evaluate(program, i)
        score += total_fitness(images, o)

    return score


def solve(task, seconds):
    start_time = time.time()
    num_programs = 10
    programs = [build(1, []) for _ in range(num_programs)]
    best_score = np.inf
    while time.time() - start_time < seconds:
        scores = []

        for p in programs:
            scores.append(evaluate_fitness(p, task['train']))
        # print(scores)

        min_score = min(scores)

        if min_score == 0:
            best_p = programs[scores.index(min_score)]
            final_score = evaluate_fitness(best_p, task['test'])
            if final_score == 0:
                print(time.time() - start_time)
                return 10
            else:
                return 1

        if min_score < best_score:
            start_time = time.time()
            best_p = programs[scores.index(min_score)]
            best_score = min_score
            # print(best_score)
            # for i in range(len(task['train'])):
            #     input_im = np.array(task['train'][i]['input'])
            #     output_im = np.array(task['train'][i]['output'])
            #     plot_grids([input_im, evaluate(best_p, input_im), output_im])

        score_prog = zip(scores, programs)
        programs = [x for _, x in sorted(score_prog, key=itemgetter(0))][:5]

        new_programs = []

        for i, p in enumerate(programs):
            new_programs.append(build(1, p))
            new_programs.append(build(1, []))

        programs = new_programs
    return 0


if __name__ == "__main__":
    training_dir = "./data/training"
    training_files = sorted(listdir(training_dir))

    if os.path.isfile('success.txt'):
        with open('success.txt', 'r+') as f:
            success_ids = '\n'.split(f)


    tasks = []
    for task_file in training_files:
        with open("/".join([training_dir, task_file])) as f:
            task_id = task_file.split(".")[0]
            tasks.append((task_id, json.load(f)))

    solves = 0
    for i, task in enumerate(tasks[:]):
        solves += solve(task[1], 5)
        if solves:
            with open('success.txt', 'a+') as f:
                f.write(task_id + '\n')
        print(i, solves)


# train000 = [upsample_c(3, 3), tile_c(3, 3), logical_and_c]
# train001 = [enclosed, pipe_c(), color_map_c()]
# train002 = [get_period_length, clip_c(), wrap_c(), change_color_c()]

