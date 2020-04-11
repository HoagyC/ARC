import json
import numpy as np
from tqdm import tqdm
from os import listdir

training_dir = "./data/training"
training_files = listdir(training_dir)

tasks = []

for task_file in training_files:
    with open("/".join([training_dir, task_file])) as f:
        tasks.append(json.load(f))

def get_output_shapes(task):
    test_outputs = [np.array(f['output']) for f in task['test']]
    shapes = [f.shape for f in test_outputs]
    return(shapes)

def guess_output_shapes(task):
    outputs = [np.array(f['output']) for f in task['train']]
    shapes = [f.shape for f in outputs]
    if len(set(shapes)) == 1:
        return [shapes[0]]*len(task['test'])
    else:
        shape_pairs = get_shape_pairs(task)
        print(shape_pairs)
        if all([a == b for a, b in shape_pairs]):
            return [get_shape(t['input']) for t in task['test']]



def get_shape_pairs(task):
    pairs = []
    for t in task['train']:
        pairs.append((get_shape(t['input']), get_shape(t['output'])))
    return pairs


def get_shape(t):
    return np.shape(np.array(t))

right_count = 0
for t in tasks:
    guess_outputs = guess_output_shapes(t)
    target_outputs = get_output_shapes(t)
    print(guess_outputs, target_outputs)

    if guess_outputs == target_outputs:
        right_count += 1

print(right_count, right_count/len(tasks))

