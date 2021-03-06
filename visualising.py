import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors

from graphviz import Digraph


if __name__ == "__main__":
    data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'

    training_tasks = sorted(os.listdir(training_path))
    evaluation_tasks = sorted(os.listdir(evaluation_path))


def plot_grid(grid):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(grid))])
    ax.set_xticks([x-0.5 for x in range(1+len(grid[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()


def plot_grids(grids):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    fig, ax = plt.subplots(1, len(grids))
    for i in range(len(grids)):
        ax[i].imshow(grids[i], cmap=cmap, norm=norm)
        ax[i].grid(True, which='both', color='lightgrey', linewidth=0.5)
        ax[i].set_yticks([x - 0.5 for x in range(1 + len(grids[i]))])
        ax[i].set_xticks([x - 0.5 for x in range(1 + len(grids[i][0]))])
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    plt.show()


def plot_one(ax, i, train_or_test, input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_one(axs[0,i],i,'train','input')
        plot_one(axs[1,i],i,'train','output')        
    plt.tight_layout()
    plt.show()        
        
    num_test = len(task['test'])
    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
    if num_test==1: 
        plot_one(axs[0],0,'test','input')
        plot_one(axs[1],0,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(axs[0,i],i,'test','input')
            plot_one(axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show()


def show_graph(prog):
    g = Digraph()

    # Adding edges
    c = 0

    def add_tree(c, prog):
        current, current_val = prog[c], c
        g.node(str(c), str(current))
        print(c)
        c += 1
        if type(current) == int:
            return c

        if current.numbers:
            for _ in range(current.numbers):
                g.edge(str(current_val), str(c))
                c = add_tree(c, prog)

        if current.grids:
            for _ in range(current.grids):
                g.edge(str(current_val), str(c))
                c = add_tree(c, prog)
        return c

    while c < len(prog):
        c = add_tree(c, prog)

    g.render('test-output/round-table.gv', view=True)
