"""
File contains:
    -> Full dataset splitting function fot train/val/split
    -> Training stats collection function
    -> Stats file search and loader
    -> Basic normalisation functions
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import math
import random
import numpy as np
import pandas as pd

from dataset_.stat_recorder_class import StatsRecorder
from dataset_.BasicDatasetClasses import FastDataLoader, BasicTrainingSet

###############################################################################
# CLASS SPLITTING FUNCTIONS
###############################################################################
def class_split(path, weight_list):
    """Grabs the available classes and splits them into train, validation and
        test sets

    Args:
        path (str): The exact path to teh dataste location, has assumed structure 
            of folders of classes 
        weight_list (list): The list of weights corresponding to the split ratios 
            of the full dataset, train/validation/test

    Returns:
        list: List of sublists containing all of the split wise classes
    """
    # Grabs a list of all classes and shuffles it
    classes = os.listdir(path)
    random.shuffle(classes)

    if sum(weight_list) > 1.0:
        raise ValueError("""WAIT THATS ILLEGAL: Weights cannot sum to more than 1, 
                this will create overlapping sets""")

    sublists = []
    prev_index = 0
    for weight in weight_list:
        next_index = prev_index + math.ceil( (len(classes) * weight) )

        sublists.append( classes[prev_index:next_index] )
        prev_index = next_index

    print(f'Total Classes: {len(classes)}')
    print(f'Train/Val/Test :: {len(sublists[0])}/{len(sublists[1])}/{len(sublists[2])}')

    return sublists

def esc_class_split(path, weight_list, groupings_path):
    """Grabs the available classes and splits them into train, validation and
        test sets. Does some specific major element groupings for ESC-50 dataset.
        This is mainly in the form of stratification between all splits. 

    Args:
        path (str): The exact path to teh dataste location, has assumed structure 
            of folders of classes 
        weight_list (list): The list of weights corresponding to the split ratios 
            of the full dataset, train/validation/test

    Returns:
        list: List of sublists containing all of the split wise classes
    """
    # Grabs a list of all classes and shuffles it
    all_classes = os.listdir(path)

    # Gets the new meta data path
    #path_sections = path.split("/")[:-1]
    #master_path = ''
    #for i in path_sections:
    #    master_path += str(i) + "/"
    #groupings_path = os.path.join(master_path + 'meta', 'major_groupings.csv')
    # Reads in major groups daraframe
    groupings = pd.read_csv(groupings_path)

    # Grabs all unique major groups 
    unique_majors = np.unique(groupings['major_group'])
    # Gets all the class groupings and puts them in lists
    class_groups = []
    for group in unique_majors:
        classes = groupings[groupings['major_group'] == group]['class'].values
        class_groups.append(classes)

    for classes in class_groups:
        random.shuffle(classes)

    if sum(weight_list) > 1.0:
        raise ValueError("""WAIT THATS ILLEGAL: Weights cannot sum to more than 1, 
                this will create overlapping sets""")

    # Test, val and test sublists
    sublists = [[], [], []]
    for classes in class_groups:
        # Can iterate over classes
        it = iter(classes)
        sizes = [math.ceil(len(classes)*weight) for weight in weight_list]
        splits = [[next(it) for _ in range(size)] for size in sizes]
        # Filter the splits into sublists
        for i, _ in enumerate(splits):
            sublists[i] += splits[i]

    print(f'Total Classes: {len(all_classes)}')
    print(f'Train/Val/Test :: {len(sublists[0])}/{len(sublists[1])}/{len(sublists[2])}')

    return sublists

###############################################################################
# TRAINING STATS GRABBER FUNCTION  
###############################################################################
def gen_training_stats(data_path, training_classes, norm, type, save_path, num_workers):
    """[summary]

    Args:
        data_path ([type]): [description]
        training_classes ([type]): [description]
        norm ([type]): [description]
        type ([type]): [description]
        save_path ([type]): [description]
        num_workers ([type]): [description]

    Returns:
        [type]: [description]
    """
    dataset = BasicTrainingSet(data_path, training_classes)
    dataloader = FastDataLoader(dataset, num_workers=num_workers, batch_size=10000)
    
    # We create different stats recorders based on what norm and data type we use
    # Raw data has shape (batch, audio_length)
    # Spec has shape (batch, channel(n_mels), time)
    if norm == 'global':
        if type == 'raw':
            # For raw data we can only do global norm
            stats_class = StatsRecorder(red_dims=(0, 1))
        # Global spec collapses all dimensions
        elif type == 'spec':
            stats_class = StatsRecorder(red_dims=(0, 1, 2))
    # Spec per channel only collapses batches and the time dimension
    elif norm == 'channel':
        stats_class = StatsRecorder(red_dims=(0, 2))
    
    # Iterates over the train dataet split and collects statistics
    for idx, (x, y) in enumerate(dataloader):
        stats_class.update(x)

    mean, std = np.array(stats_class.mean[0]), np.array(stats_class.std[0])
    joint_stats = np.array([mean, std])
    np.save(save_path, joint_stats)

    return joint_stats

###############################################################################
# CHECK FOR RELEVANT STATS
###############################################################################
def check_for_stats(folder_name, file_name):
    """Searches the stats collection folder and tries to find a relevent stats
        file.

    Args:
        folder_name (str): The exact path of the stats folder
        file_name (str): The name of the stats file being searched for

    Returns:
        None or array: [description]
    """
    try:
        files = os.listdir(folder_name)

    except Exception:
        os.mkdir(folder_name)
        return [None, None]

    for file in files:
        if file_name in file:
            print(f'Stats file found: {file_name}')
            stats = np.load( os.path.join(folder_name, file_name) )
            return stats

    return [None, None]

##############################################################################
# NORMALISATION/SCALING FUNCTIONS
##############################################################################
def nothing_func(x):
    return x

def per_sample_scale(x):
    return (x- x.mean()) / x.std()

def given_stats_scale(x, mu, sigma):
    return (x - mu) / sigma
