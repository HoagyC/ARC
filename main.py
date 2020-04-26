import numpy as np
import random
from os import listdir
import os
import json
import time
import copy
from operator import itemgetter

from visualising import plot_grids
from gridRL import pad_grid, GridRL


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


def build(prog_len=1, start=[]):
    functions = [identity, identity, identity, swap_color, tile, upsample, color_map, enclosed, clip, logical_and, wrap, select_col,
                 flip, rotate, crop, rectangulate, not_, concath, concatv]
    points = [(0, 0)]
    numbers = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, count_col, count, top_col]

    def add_function():
        add = random.choice(functions)

        new_fn = [add, add_args(add)]

        return new_fn

    def add_args(func):
        add_nums, add_points, add_grids = [], [], []
        if func.numbers:
            nums = random.choices(numbers, k=func.numbers)
            for n in nums:
                if type(n) == int:
                    add_nums.append(n)
                else:
                    add_nums.append([n, add_args(n)])

        if func.points:
            add_points = random.choices(points, k=func.points)

        if func.grids:
            add_grids.append(add_function())

        return [add_nums, add_points, add_grids]

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
        if i.shape[0] == 0 or i.shape[1] == 0:
            return np.inf
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
            print(min_score)
            start_time = time.time()
            best_p = programs[scores.index(min_score)]
            best_score = min_score

        score_prog = sorted(zip(scores, programs), key=itemgetter(0))
        programs = [x for _, x in score_prog][:5]
        new_programs = []
        for i, p in enumerate(programs):
            new_programs.append(p)
            new_p = copy.copy(p)
            if random.random() > 0.5:
                new_programs.append(build(1, new_p))
            else:
                new_programs.append(build(1, []))

        programs = new_programs

    best_p = programs[scores.index(min_score)]
    if min_score < 5:
        print(evaluate_fitness(best_p, task['test']))

    input_im = np.array(task['train'][1]['input'])
    output_im = np.array(task['train'][1]['output'])
    plot_grids([input_im, evaluate(best_p, input_im), output_im])

    return 0


if __name__ == "_main__":
    training_dir = "./data/training"
    training_files = sorted(listdir(training_dir))

    if os.path.isfile('success.txt'):
        with open('success.txt', 'r+') as f:
            ids = f.read()
            success_ids = ids.splitlines()
    else:
        success_ids = []

    tasks = []
    for task_file in training_files:
        with open("/".join([training_dir, task_file])) as f:
            task_id = task_file.split(".")[0]
            tasks.append((task_id, json.load(f)))

    task = tasks[1][1]
    # train001 = [wrap, [[1, [[swap_color, [[2, 4], [], []]]], [], []]]]
    # print(evaluate_fitness(train001, task['train']))
    # print(evaluate_fitness(train001, task['test']))
    input_im = np.array(task['train'][1]['input'])
    g = GridRL([], [1])
    print(g(pad_grid(input_im), []).shape)
    output_im = np.array(task['train'][1]['output'])
    plot_grids([input_im, pad_grid(input_im)])

    solves = 0
    for i, task in enumerate(tasks):
        score = solve(task[1], 30)
        solves += score
        if score and task[0] not in success_ids:
            with open('success.txt', 'a+') as f:
                f.write(str(i) + ' ' + task[0] + '\n')
                print('NEW SOLVE WOOOOOOOOOOO')
        print(i, solves)


train000 = [upsample, [[3, 3], [], []], logical_and, [[], [], [[tile, [[3, 3], [], []]]]]]
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
# train001 = [colormap, [[4], [], [[enclosed, [], [], []]]]]
