"""
File contains:
    -> Full dataset splitting function fot train/val/split
    -> Custom collation function for variable length sets 
    -> Stats file search and loader
    -> Training stats collection function
    -> Mel-spec function
    -> Basic normalisation functions
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import math
import torch
import random
import librosa
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

###############################################################################
# VARIABLE LENGTH TRAINING STATS COLLATE FUNCTION
###############################################################################
def my_collate(batch):
    # Including this empty part in array causes issues
    if batch[0][0].ndim < 2:
        samples = torch.empty(size=(1, batch[0][0].shape[-1]))
        raw = True
    elif batch[0][0].ndim == 2:
        samples = torch.empty(size=(1, batch[0][0].shape[-1]))
        raw = False
    else:
        samples = torch.empty(size=(1, batch[0][0].shape[-2], batch[0][0].shape[-1]))
        raw = False

    for set in batch:
        x, y = set

        if raw:
            x = x.unsqueeze(0)

        samples = torch.cat((samples, x),dim=0)
    #print(torch.mean(samples))
    # Tragets is wrong shape as not multiplied up but it isnt used anyway 
    target = [item[1] for item in batch]
    return [samples[1:], target]

###############################################################################
# TRAINING STATS GRABBER FUNCTION  
###############################################################################
def gen_training_stats(data_path, training_classes, norm, type, save_path, num_workers, params):
    """Generates the training set mean and std for normalisation purposes

    Args:
        data_path (str): The exact path to the expected data
        training_classes (list): List of classes to use in iteration
        norm (str): Type of normalisation we wish to calculate 
        type (str): The type of data being input, 'spec/raw/rawtospec
        save_path (str): The path to save teh normlaisation stats to
        num_workers (int): Number of workers to use in the dataloader class
        params (dict): Full experiment paramater dictionary

    Returns:
        array: Nested array of mean and std of training set: mean, std = []
    """
    #return [0 ,0.1]
    dataset =  BasicTrainingSet(data_path, training_classes)
    if type in ['spec', 'raw']:
        dataloader = FastDataLoader(dataset, num_workers=num_workers, batch_size=500)
    elif type in ['variable_spec', 'variable_raw']:
        dataloader = FastDataLoader(dataset, num_workers=num_workers, collate_fn=my_collate ,
                                batch_size=500)
    
    # We create different stats recorders based on what norm and data type we use
    # Raw data has shape (batch, audio_length)
    # Spec has shape (batch, channel(n_mels), time)
    # Raw to spec has final shape (batch, channel(n_mels), time), needs diff loader
    if norm == 'global':
        if type in ['raw', 'variable_raw']:
            # For raw data we can only do global norm
            stats_class = StatsRecorder(red_dims=(0, 1))
        # Global spec collapses all dimensions
        elif type in ['spec', 'variable_spec']:
            stats_class = StatsRecorder(red_dims=(0, 1, 2))
    # Spec per channel only collapses batches and the time dimension
    elif norm == 'channel':
        stats_class = StatsRecorder(red_dims=(0, 2))
    
    # Iterates over the train dataet split and collects statistics
    for idx, (x, y) in enumerate(dataloader):
        stats_class.update(x)
        #print(stats_class.mean, stats_class.std)

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
# MEL-SPECTROGRAM FUNCTION
##############################################################################
def mel_spec_function(x, sr, n_mels, n_fft, hop_length, power):
    mel_spec_array = librosa.feature.melspectrogram(y=x,
                                    sr=sr,
                                    n_mels=n_mels,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    power=power)

    log_mel_spec = 20.0 / power * np.log10(mel_spec_array + sys.float_info.epsilon)
    return log_mel_spec
##############################################################################
# NORMALISATION/SCALING FUNCTIONS
##############################################################################
def nothing_func(x):
    return x

def per_sample_scale(x):
    return (x- x.mean()) / x.std()

def given_stats_scale(x, mu, sigma):
    return (x - mu) / sigma
