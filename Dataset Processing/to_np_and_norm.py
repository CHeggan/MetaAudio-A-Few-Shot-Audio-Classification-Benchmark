"""
Script deals with:
    -> Converson of wav files to npy, faster loading and processing etc with
        reduced storage size
    -> Per data sample z normalisation

The main function iterates over the so called 'old_dir'and creates a mirror
    directory in 'new_dir'.

We assume that all samples are valid length an std and none have to be
    removed. 
"""
###############################################################################
# IMPORTS
###############################################################################
from tqdm import tqdm

import numpy as np
import librosa
import yaml
import time
import sys
import os

##############################################################################
# HELPER FUNCTIONS
##############################################################################
def normalise(data):
    """Performs standard z normalisation to the input data

    Args:
        data (array): The data sample array, assumed to work with numpy like
            operations

    Returns:
        array: The newly normalised data sample
    """
    # Performs per-sample normaisation to mean 0 and std 1
    new_data = ((data - np.mean(data)) / np.std(data))
    return new_data

def wav_to_npy(path, sr):
    """Function loads some raw audio file from path using librosa library and
        then converts it to a numpy array before returning

    Args:
        path (str): The exact path of the raw audio file, including the '.wav'
        sr (int): The sample rate to load the data in at

    Returns:
        array: The raw audio signal in numpy format
    """
    data, sr = librosa.load(path, sr=sr, mono=True)
    data = np.array(data)
    return data

###############################################################################
# CONVERSION FUNCTION
###############################################################################
def file_conversion(new_dir, current_path, sr, norm):
    """Takes some path for a raw audio file and carries out the full loading,
        conversion and normalisation before saving it to a .npy file

    Args:
        new_dir (str): The path to the new directory, where the file being
            loaded should eventually be saved
        current_path (str): The exact path of the raw audio file
        sr (int): The sample rate to load the data in at
        norm(boolean): Whether we should per sample normalise the data
    """
    # Loads the current data sample and then converts to numpy
    data = wav_to_npy(current_path, sr)

    if norm == True:
        # z normalises the data
        data = normalise(data)

    file_name = current_path.split('\\')[-1]
    file_name = file_name.split('.')[0]

    new_path = os.path.join(new_dir, file_name)
    #print(new_path, np.mean(new_data), np.std(new_data))

    # Saves the newly normalised sample to new directory
    np.save(new_path, data)

###############################################################################
# MAIN FUNCTION
###############################################################################
def main(old_dir, new_dir, sr, norm):
    """Iterates thorugh the folder structure of old directory and mirrors it in the
        new dir folder with per-sample normalised data which has been converted 
        from .wav to .npy.

    Args:
        old_dir (str): The current raw data directory we would like to mirror 
            with processed data 
        new_dir (str): The path to the directory we would like to use to store 
            our new processed data
        sr (int): The sample rate to convert the data samples into
        norm (boolean): Whether to apply per-sample norm to the data 
    """
    try:
        os.mkdir(new_dir)

    except Exception:
        print(f'Cannot create folder: {new_dir}')

    #sets working dir as array parent folder
    os.chdir(new_dir)

    # Gets the names of the included classes
    contained_classes = os.listdir(old_dir)

    # Iterates over the full folder structure and counts how many wav files 
    total_files = 0
    for root, _, files in os.walk(old_dir):
        for f in files:
            if f.endswith('.wav'):
                total_files += 1
    
    # A little visual to see how the processing is coming along
    progress_bar = tqdm(total=total_files)

    for seen_class in contained_classes:
            working_dir = os.path.join(old_dir, seen_class)
            new_working_dir = os.path.join(new_dir, seen_class)

            try:
                os.mkdir(new_working_dir)
            except Exception:
                print(f'Already Created: {new_working_dir}')

            for fname in os.listdir(working_dir):
                if fname.endswith('.wav'):
                    file_path = os.path.join(working_dir, fname)

                    file_conversion(new_working_dir, file_path, sr, norm)
                    # Update the progress bar
                    progress_bar.update(1)

###############################################################################
# MAIN CALL
###############################################################################
# Laptop
#old_dir = 'C:/Users/user/Documents/Datasets/FSD AudioSet Small/Sorted_raw'
#new_dir = 'C:/Users/user/Documents/Datasets/FSD AudioSet Small/Sorted_npy'

# Desktop
#old_dir = 'X:/Datasets/ESC-50-master/Sorted'
#new_dir = 'X:/Datasets/ESC-50-master/Sorted_npy'

#main(old_dir, new_dir, 16000)
