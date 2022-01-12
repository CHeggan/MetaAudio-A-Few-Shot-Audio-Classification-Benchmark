"""
Utility and helper functions for our distribuion search for variable length.

Contains:
    -> Split generator
    -> Load file and get length
    -> Extract KDE
    -> Histogram and distribuion plotter    
    -> Distribution / basic line Plotter

"""
################################################################################
# IMPORTS
################################################################################
import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

################################################################################
# FUNCTIONS 
################################################################################
def extract_kde(length_values, gridsize=200):
    my_kde = sns.kdeplot(length_values, gridsize=gridsize)
    line = my_kde.lines[0]
    x, y = line.get_data()
    x, y = np.array(x), np.array(y)
    plt.clf()
    return x, y


def load_and_length(data_path, sr):
    data = np.load(data_path)
    length = float(data.shape[0]/sr)
    return length 


def class_split(classes, weight_list, verbose=False):
    if sum(weight_list) > 1.0:
        raise ValueError("""WAIT THATS ILLEGAL: Weights cannot sum to more than 1, 
                this will create overlapping sets""")

    sublists = []
    prev_index = 0
    for weight in weight_list:
        next_index = prev_index + math.ceil( (len(classes) * weight) )

        sublists.append( classes[prev_index:next_index] )
        prev_index = next_index

    if verbose:
        print(f'Total Classes: {len(classes)}')
        print(f'Train/Val/Test :: {len(sublists[0])}/{len(sublists[1])}/{len(sublists[2])}')

    return sublists


def save_splits(sub_lists, dataset_name, type):
    """Saves and outputs the class splits generated from fixed seed

    Args:
        sub_lists (list of lists): A list of lists of classes, [train, val, test]
        dataset_name (str): The name of the dataset being split up 
    """
    sub_lists = np.array(sub_lists, dtype='object')
    sub_dict = {'Training Classes':sub_lists[0], 'Validation Classes': sub_lists[1],
        'Testing Classes': sub_lists[2]}
    df = pd.DataFrame.from_dict(sub_dict, orient='index')

    np.save(dataset_name + '_' + str(type) + '_split.npy', sub_lists)
    df.to_csv(dataset_name + '_' + str(type) + '_split.csv')
    print('Classes split generated')

################################################################################
# PLOTTING
################################################################################
def plot_dist(length_values, bins, title, save=False):
    sns.histplot(length_values, bins=bins, kde=True)
    plt.xlabel('Length in Seconds')
    plt.ylabel('Instances')
    plt.title(title)
    if save:
        plt.savefig(title + '.svg')
    plt.show()


def plot_line(x, y):
    plt.plot(x, y)
    plt.show()
