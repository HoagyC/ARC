import numpy as np
import random
from os import listdir
import os
import json
import time
import copy
from operator import itemgetter

from visualising import plot_grids
from gridRL import pad_grid, GridRL, flatten, build_flat_model
from DSL_functions import get_func_list, get_num_list, evaluate, logical_and, tile, upsample, evaluate_flat, color_map, enclosed, rectangulate

import torch


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
        images = evaluate_flat(program, i)
        score += total_fitness(images, o)

    return score


def solve(task, seconds, model, temp):
    start_time = time.time()
    best_score = np.inf
    best_program = None

    in_image = np.array(task['train'][0]['input'])
    in_image = pad_grid(in_image)
    out_image = np.array(task['train'][0]['output'])
    out_image = pad_grid(out_image)

    while time.time() - start_time < seconds:

        program = build_flat_model(model, in_image, out_image, temp=temp)
        while len(program) > 40:
            program = build_flat_model()

        score = evaluate_fitness(program, task['train'])

        if score < best_score:
            best_score = score
            best_program = program
            start_time = time.time()

        if best_score == 0:
            final_score = evaluate_fitness(best_program, task['test'])
            print(time.time() - start_time, best_program)

            if final_score == 0:
                return 10, final_score
            else:
                return 1, final_score

    if best_score < 5:
        print('got close, scored ', evaluate_fitness(best_program, task['test']))

    # input_im = np.array(task['train'][1]['input'])
    # output_im = np.array(task['train'][1]['output'])
    # plot_grids([input_im, evaluate_flat(best_program, input_im), output_im])

    final_score = evaluate_fitness(best_program, task['test'])
    print(best_program)
    return 0, final_score


if __name__ == "__main__":

    # Setting paths for getting files.
    training_dir = "./data/training"
    training_files = sorted(listdir(training_dir))

    # Getting task ids for tasks already solved.
    if os.path.isfile('success.txt'):
        with open('success.txt', 'r+') as f:
            ids = f.read()
            success_ids = ids.splitlines()
    else:
        success_ids = []

    # Getting all tasks and compiling into a list.
    tasks = []
    for task_file in training_files:
        with open("/".join([training_dir, task_file])) as f:
            task_id = task_file.split(".")[0]
            tasks.append((task_id, json.load(f)))

    # Setting arguments for solving the model.
    model_path = 'models/20200508.201046'
    wait_time = 5



    # Setting up model.
    func_list = get_func_list()
    num_list = get_num_list()
    model = GridRL(num_list, func_list)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # in_im = np.array(tasks[1][1]['train'][0]['input'])
    # x = evaluate_flat(train001, in_im)
    # plot_grids([in_im, x])

    # Attempting to solve each task in turn.

    temps = [1, 10, 100, 1000]

    for temp in temps:
        solves = 0
        scores = []
        for i, task in enumerate(tasks[:80]):
            complete, score = solve(task[1], wait_time, model, temp)
            solves += complete
            scores.append(score)

            # Notifying if a previously unsolved task is completed.
            if complete and task[0] not in success_ids:
                with open('success.txt', 'a+') as f:
                    f.write(str(i) + ' ' + task[0] + '\n')
                    print('NEW SOLVE WOOOOOOOOOOO')
            print(i, solves, np.mean(np.sqrt(scores)))


# train000 = [upsample, 3, 3, logical_and, tile, 3, 3]
# train001 = [colormap, [[4], [], [[enclosed, [], [], []]]]]
