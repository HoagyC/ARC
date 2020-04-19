import json
import itertools
from os import listdir
import random

import numpy as np

import manual_coding
from visualising import plot_grids

transforms = []


# Functions written by me

def select(grid, x, y, x_size, y_size):
    assert x + x_size <= grid.shape[0]
    assert y + y_size <= grid.shape[1]
    return grid[x : x + x_size, y : y + y_size]


def paste(grid, subgrid, x, y):
    assert subgrid.shape[0] + x <= grid.shape[0]
    assert subgrid.shape[1] + x <= grid.shape[0]
    grid[x : x + subgrid.shape[0], y : y + subgrid.shape[1]] = subgrid
    return grid


def color_square(x, y, color):
    return np.full((x, y), color)


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
                    end = start + 1
                else:
                    if all(r[start:]):
                        end = len(r)
                    else:
                        end = np.argmin(r[start:]) + start
                start_row = i

        recs.append([start_row, start, rows, end - start])
        bool_grid[start_row : start_row + rows, start:end] = False
    return recs


def draw_rec(grid, x1, x2, y1, y2, color):
    rec = color_square(abs(x1 - x2) + 1, abs(y1 - y2) + 1, color)
    return paste(grid, rec, min(x1, x2), min(y1, y2))


def get_difference(g1, g2):
    assert g1.shape == g2.shape
    return g1 != g2


def accuracy(t_pred, t_out):
    # compares matching % of two grids
    if t_pred.size == t_out.size:
        return (t_pred == t_out).sum() / t_out.size
    else:
        return 0


def select_points(grid, select):
    points = np.isin(grid, select)
    return np.argwhere(points)


def freq(grid):
    return [(grid == i).sum() for i in range(10)]


def get_corners(grid):
    pass


def mismatches(a, b):
    return a == b


def low_colours(g, freq):
    low = min([f for f in freq if f > 0])
    return [c for c, n in enumerate(freq) if n == low]


def match_bool(a, b):
    x, y = min(a.shape, b.shape)
    return a[:x, :y] == b[:x, :y]


def build_output(in_grid, out_grid, rec_list):
    out_build = np.zeros(out_grid.shape, dtype=bool)
    yes_recs = []
    rec_list.sort(key=lambda x: -x[2] * x[3])
    out_x, out_y = out_grid.shape
    while not out_build.all():
        for rec_id, rec in enumerate(rec_list):
            if rec[2] <= out_x and rec[3] <= out_y:
                for i in range(out_x - rec[2] + 1):
                    for j in range(out_y - rec[3] + 1):
                        match = (
                            out_grid[i : i + rec[2], j : j + rec[3]]
                            == in_grid[
                                rec[0] : rec[0] + rec[1], rec[2] : rec[2] + rec[3]
                            ]
                        )
                        new = out_build[i : i + rec[2], j : j + rec[3]]
                        is_new = (type(new) == bool and new == False) or (
                            type(new) != bool and not new.all()
                        )
                        is_match = (type(match) == bool and match == True) or (
                            type(match) != bool and match.all()
                        )
                        if is_new and is_match:
                            yes_recs.append([i, j, rec_id])
                            out_build[i : i + rec[2], j : j + rec[3]] = True
                            plot_grid(out_build)
                            print(yes_recs)
    return yes_recs


# Functions from the notebook DSL


# np.array -> [np.array]
def groupByColor_unlifted(pixmap):
    """ Split an image into a collection of images with unique color """
    # Count the number of colors
    nb_colors = int(pixmap.max()) + 1
    # Create a pixmap for each color
    splited = [(pixmap == i) * i for i in range(1, nb_colors)]
    # Filter out empty images
    return [x for x in splited if np.any(x)]


# np.array -> [np.array]
def cropToContent_unlifted(pixmap):
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(pixmap)
    if len(true_points) == 0:
        return []
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    pixmap = pixmap[
        top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1
    ]
    return [pixmap]


# np.array -> [np.array]
def splitH_unlifted(pixmap):
    """ Split horizontally an image """
    h = pixmap.shape[0]
    if h % 2 == 1:
        h = h // 2
        return [pixmap[:h,:], pixmap[h+1:,:]]
    else:
        h = h // 2
        return [pixmap[:h,:], pixmap[h:,:]]


# np.array -> [np.array]
def negative_unlifted(pixmap):
    """ Compute the negative of an image (and conserve the color) """
    negative = np.logical_not(pixmap).astype(int)
    color = max(pixmap.max(), 1)
    return [negative * color]


# [np.array] -> [np.array]
def identity(x: [np.array]):
    return x


# [np.array] -> [np.array]
def tail(x):
    if len(x) > 1:
        return x[1:]
    else:
        return x


# [np.array] -> [np.array]
def init(x):
    if len(x) > 1:
        return x[:1]
    else:
        return x


# [np.array] -> [np.array]
def union(x):
    """ Compute the pixel union of all images in the list. """
    if len(x) < 2:
        return x

    # Make sure everybody have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []

    return [np.bitwise_or.reduce(np.array(x).astype(int))]


def intersect(x):
    """ Compute the pixel intersection of all images in the list. """
    if len(x) < 2:
        return x

    # Make sure everybody have the same shape
    first_shape = tuple(x[0].shape)
    for pixmap in x[1:]:
        if first_shape != tuple(pixmap.shape):
            return []

    return [(np.prod(np.array(x), axis=0) > 0).astype(int)]


def sortByColor(xs):
    """ Sort pictures by increasing color id. """
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: x.max()))


def sortByWeight(xs):
    """ Sort images by how many non zero pixels are contained. """
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: (x > 0).sum()))


def reverse(x):
    """ Reverse the order of a list of images. """
    return x[::-1]


def lift(fct):
    # Lift the function
    def lifted_function(xs):
        list_of_results = [fct(x) for x in xs]
        return list(itertools.chain(*list_of_results))
    # Give a nice name to the lifted function
    import re
    lifted_function.__name__ = re.sub('_unlifted$', '_lifted', fct.__name__)
    return lifted_function


cropToContent = lift(cropToContent_unlifted)
groupByColor = lift(groupByColor_unlifted)
splitH = lift(splitH_unlifted)
negative = lift(negative_unlifted)


def is_solution(program, task):
    for sample in task:  # For each pair input/output
        i = np.array(sample['input'])
        o = np.array(sample['output'])

        # Evaluate the program on the input
        images = evaluate(program, i)
        if len(images) < 1:
            return False

        # The solution should be in the 3 first outputs
        images = images[:3]

        # Check if the output is in the 3 images produced
        is_program_of_for_sample = any([np.array_equal(x, o) for x in images])
        if not is_program_of_for_sample:
            return False

    return True


def width_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right width. Less is better."""
    return np.abs(predicted.shape[0] - expected_output.shape[0])


def height_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right height. Less is better."""
    return np.abs(predicted.shape[1] - expected_output.shape[1])


def activated_pixels_fitness(p, e):
    """ How close the predicted image to have the right pixels. Less is better."""
    shape = (max(p.shape[0], e.shape[0]), max(p.shape[1], e.shape[1]))
    diff = np.zeros(shape, dtype=int)
    diff[0:p.shape[0], 0:p.shape[1]] = (p > 0).astype(int)
    diff[0:e.shape[0], 0:e.shape[1]] -= (e > 0).astype(int)

    return (diff != 0).sum()


def colors_fitness(p, e):
    p_colors = np.unique(p)
    e_colors = np.unique(e)

    nb_inter = len(np.intersect1d(p_colors, e_colors))

    return (len(p_colors) - nb_inter) + (len(e_colors) - nb_inter)


def total_fitness(p, e):
    shape = (min(p.shape[0], e.shape[0]), min(p.shape[1], e.shape[1]))
    incorrect = (p[0:shape[0], 0:shape[1]] != e[0:shape[0], 0:shape[1]]).sum()
    misshape = abs(p.shape[0]-e.shape[0]) * abs(p.shape[1] + e.shape[1])
    return incorrect + misshape


fitness_functions = [total_fitness]


def product_less(a, b):
    """ Return True iff the two tuples a and b respect a<b for the partial order. """
    a = np.array(a)
    b = np.array(b)
    return (np.array(a) < np.array(b)).all()


# ([[np.array] -> [np.array]], Task) -> (int, int, ..., int)
def evaluate_fitness(program, task):
    """ Take a program and a task, and return its fitness score as a tuple. """
    score = 0

    # For each sample
    for sample in task:
        i = np.array(sample['input'])
        o = np.array(sample['output'])

        # For each fitness function
        images = evaluate(program, i)
        score += total_fitness(images[0], o)

    return score


def new_build():
    functions = [identity]
    points = [(0,0)]
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    grids = [identity]

    for i in range(5):
        def random_function():
            return random.choice(functions)

        new_fn = []

        add = random_function()
        if add.numbers:
            add_nums = random.choices(numbers, k=add.numbers)

        if add.points:
            add_points = random.choices(points, k=add.points)

        if add.grids:
            add_grids = [[random_function()]]

        new_fn += [add, add_nums, add_points, add_grids]

    return new_fn


print(new_build())


def new_evaluate(program: [], input_image: np.array):
    # Make sure the input is a np.array
    input_image = np.array(input_image)
    assert type(input_image) == np.ndarray

    # Apply each function on the image
    image_list = [input_image]
    for fct in program:
        # Apply the function
        image_list = fct(image_list)
        # Filter out empty images
        image_list = [img for img in image_list if img.shape[0] > 0 and img.shape[1] > 0]
        # Break if there is no data
        if image_list == []:
            return []
    return image_list


def build_candidates(allowed_nodes=[identity], best_candidates=[], nb_candidates=200):
    """
    Create a poll of fresh candidates using the `allowed_nodes`.

    The pool contain a mix of new single instructions programs
    and mutations of the best candidates.
    """
    new_candidates = []
    length_limit = 4  # Maximal length of a program
    def random_node():
        return random.choice(allowed_nodes)

    # Until we have enough new candidates
    while (len(new_candidates) < nb_candidates):
        # Add 10 new programs
        for i in range(5):
            new_candidates += [[random_node()]]

        # Create new programs based on each best candidate
        for best_program in best_candidates:
            # Add one op on its right but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [[random_node()] + best_program]
            # Add one op on its left but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [best_program + [random_node()]]
            # Mutate one instruction of the existing program
            new_candidates += [list(best_program)]
            new_candidates[-1][random.randrange(0, len(best_program))] = random_node()

    # Truncate if we have too many candidates
    np.random.shuffle(new_candidates)
    return new_candidates[:nb_candidates]


def build_model(task, max_iterations=20, verbose=True):
    candidates_nodes = [
        tail, init, union, intersect,
        sortByColor, sortByWeight, reverse,
        cropToContent, groupByColor, splitH,
        negative
    ]

    if verbose:
        print("Candidates nodes are:", [program_desc([n]) for n in candidates_nodes])
        print()

    best_candidates = {}  # A dictionary of {score:candidate}
    for i in range(max_iterations):
        if verbose:
            print("Iteration ", i + 1)
            print("-" * 10)

        # Create a list of candidates
        candidates = build_candidates(candidates_nodes, best_candidates.values())

        # Keep candidates with best fitness.
        # They will be stored in the `best_candidates` dictionary
        # where the key of each program is its fitness score.
        for candidate in candidates:
            score = evaluate_fitness(candidate, task)
            is_incomparable = True  # True if we cannot compare the two candidate's scores

            # Compare the new candidate to the existing best candidates
            best_candidates_items = list(best_candidates.items())
            for best_score, best_candidate in best_candidates_items:
                print(score, best_score)
                if score <= best_score:
                    # Remove previous best candidate and add the new one
                    del best_candidates[best_score]
                    best_candidates[score] = candidate
                    is_incomparable = False  # The candidates are comparable
                if product_less(best_score, score) or best_score == score:
                    is_incomparable = False  # The candidates are comparable
            if is_incomparable:  # The two candidates are incomparable
                best_candidates[score] = candidate

        # For each best candidate, we look if we have an answer
        for program in best_candidates.values():
            if is_solution(program, task):
                print("???")
                return program

        # Give some information by selecting a random candidate
        if verbose:
            print("Best candidates length:", len(best_candidates))
            random_candidate_score = random.choice(list(best_candidates.keys()))
            print("Random candidate score:", random_candidate_score)
            print("Random candidate implementation:", program_desc(best_candidates[random_candidate_score]))
    return list(best_candidates.values())[0]


def program_desc(program):
    """ Create a human readable description of a program. """
    desc = [x.__name__ for x in program]
    return(' >> '.join(desc))


def evaluate(program: [], input_image: np.array):
    # Make sure the input is a np.array
    input_image = np.array(input_image)
    assert type(input_image) == np.ndarray

    # Apply each function on the image
    image_list = [input_image]
    for fct in program:
        # Apply the function
        image_list = fct(image_list)
        # Filter out empty images
        image_list = [img for img in image_list if img.shape[0] > 0 and img.shape[1] > 0]
        # Break if there is no data
        if image_list == []:
            return []
    return image_list


def solve_task(task):
    # program = build_model(task['train'], verbose=False)

    for i in range(len(task['train'])):
        # result = evaluate(program=program, input_image=task['train'][i]['input']
        task_input = np.array(task['train'][i]['input'])
        result = manual_coding.task_train001(task_input)
        plot_grids([task['train'][i]['input'], result, task['train'][i]['output']])


if __name__ == "__main__":
    training_dir = "./data/training"
    training_files = sorted(listdir(training_dir))

    tasks = []
    for task_file in training_files:
        with open("/".join([training_dir, task_file])) as f:
            task_id = task_file.split(".")[0]
            tasks.append((task_id, json.load(f)))

    solve_task(tasks[1])
