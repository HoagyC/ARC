import json
from os import listdir

import numpy as np

from visualising import plot_grid

transforms = []

def select(grid, x, y, x_size, y_size):
    assert(x + x_size <= grid.shape[0])
    assert(y + y_size <= grid.shape[1])
    return(grid[x:x+x_size, y:y+y_size])

def paste(grid, subgrid, x, y):
    assert(subgrid.shape[0] + x <= grid.shape[0])
    assert(subgrid.shape[1] + x <= grid.shape[0])
    grid[x:x+subgrid.shape[0], y:y+subgrid.shape[1]] = subgrid
    return grid

def color_square(x, y, color):
    return(np.full((x,y), color))

def rectangulate(bool_grid):
    recs = []
    # Rectangles given in form (x, y, x_len, y_len)
    while bool_grid.any():
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
                    end = start+1
                else:
                    if all(r[start:]):
                        end = len(r)
                    else:
                        end = np.argmin(r[start:]) + start
                start_row = i

        recs.append([start_row, start, rows, end - start])
        bool_grid[start_row:start_row + rows, start:end] = False
    return(recs)


def draw_rec(grid, x1, x2, y1, y2, color):
    rec = color_square(abs(x1-x2)+1, abs(y1-y2)+1, color)
    return paste(grid, rec, min(x1, x2), min(y1, y2))

def get_difference(g1, g2):
    assert(g1.shape == g2.shape)
    return g1 != g2

def accuracy(t_pred, t_out):
    #compares matching % of two grids
    if t_pred.size == t_out.size:
        return (t_pred==t_out).sum()/t_out.size
    else:
        return 0

def select_points(grid, select):
    points = np.isin(grid, select)
    return np.argwhere(points)

def freq(grid):
    return [(grid==i).sum() for i in range(10)]

def get_corners(grid):
    pass


def mismatches(a, b):
    return a == b

def low_colours(g, freq):
    low = min([f for f in freq if f>0])
    return [c for c, n in enumerate(freq) if n == low]


def build_output(in_grid, out_grid, rec_list):
    match_grid = match_bool(in_grid, out_grid)
    yes_recs = []
    rec_list.sort(key=lambda x: x[2]*x[3])
    out_x, out_y = out_grid.shape
    while not match_grid.all():
        for rec_id, rec in enumerate(rec_list):
            if rec[2] <= out_x and rec[3] <= out_y:
                for i in range(out_x-rec[2]+1):
                    for j in range(out_y - rec[3] + 1):
                        match = out_grid[i:i+rec[2], j:j+rec[3]] == in_grid[rec[0]:rec[0]+rec[1], rec[2]:rec[2]+rec[3]]

                        if (type(match) == bool and match == True) or (type(match) != bool and match.all()):
                            yes_recs.append([i, j, rec_id])
                            match_grid[i:i+rec[2], j:j+rec[3]] = True
        plot_grid(match_grid)
        print(yes_recs)
    return yes_recs

def match_bool(a,b):
    x, y = min(a.shape, b.shape)
    return a[:x,:y] == b[:x, :y]

def solve_task(task):
    train_inputs = [np.array(t['input']) for t in task[1]['train']]
    train_outputs = [np.array(t['output']) for t in task[1]['train']]

    test_inputs = [np.array(t['input']) for t in task[1]['test']]
    test_outputs = [np.array(t['output']) for t in task[1]['test']]

    mismatch_grids = [match_bool(train_inputs[i], train_outputs[1]) for i in range(len(train_inputs))]
    change_recs = [rectangulate(g) for g in mismatch_grids]
    build_output(train_inputs[0], train_outputs[0], change_recs[0])
    
    g = test_inputs[0]
    # Feature collection
    f = freq(grid)
    low_col = low_colours(g)
    rare = select_points(g, low_col)

    plot_grid(g)

if __name__ == "__main__":
    training_dir = "./data/evaluation"
    training_files = listdir(training_dir)
    
    tasks = []
    for task_file in training_files:
        with open("/".join([training_dir, task_file])) as f:
            task_id = task_file.split('.')[0]
            tasks.append((task_id,json.load(f)))

    solve_task(tasks[0])
