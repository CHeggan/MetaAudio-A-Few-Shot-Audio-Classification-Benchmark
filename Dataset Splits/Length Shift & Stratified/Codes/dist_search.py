"""
Code aims to generate length stratified and length shifted distributions for 
    class-wise splits. We approach this using a ranking table of expected values
    around the peaks of class sample distributions:
        -> Random: 
        -> Ranking: 
"""

################################################################################
# IMPORTS
################################################################################
import os 
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *
################################################################################
# STAT FUNCTIONS
################################################################################
def greedy_expected_value(all_lengths, x, y):
    """Caluclates expected value around teh peak, greedily going one bin outward
        in each direction until some % of data of that class is included

    Args:
        all_lengths (list): A list of all the sampe legths for that class
        x (list): List of x values from the kde plot
        y (list): List of y values from kde plot

    Returns:
        float: Expected average around the peak
    """
    max_idy = np.argmax(y)
    all_lengths = np.array(all_lengths)

    i = 1
    contained = 0
    # We repeat until 60% of samples in any given class are used
    while contained < 0.6:
        idx_1 = max(0, max_idy - i)
        idx_2 = min(x.shape[0], max_idy + i)

        x_low = x[idx_1]
        x_high = x[idx_2]

        contained_values = all_lengths[x_low <= all_lengths]
        contained_values = contained_values[contained_values <= x_high]

        contained = len(contained_values) / all_lengths.shape[0]
        i += 1

    x_of_care = x[idx_1:idx_2]
    y_of_care = x[idx_1:idx_2]

    # Calculate the expected value of the 60% of data we consider
    peak_ex_avg = np.sum(np.array(x_of_care) * np.array(y_of_care)) / sum(y_of_care)
    return peak_ex_avg


def stratified_sort(class_dict):
    """Sorting function to generate stratified splits. Sorts the list or dict into
        a long,short,long,short etc expected value sequence 

    Args:
        class_dict (dict): Class-wise expected average disctionary

    Returns:
        dict: 
    """
    values = list(class_dict.values())
    classes = list(class_dict.keys())
    operations = int(np.ceil(len(values)/2))

    new_list = []
    new_classes = []

    for i in range(operations):
        idx_low = 0 + i
        idx_high = - 1 -i
        idx_high_converted = len(values) + idx_high

        if idx_low == idx_high_converted:
            new_list.append(values[idx_low])
            new_classes.append(classes[idx_low])
        else:
            new_list.append(values[idx_low])
            new_list.append(values[idx_high])

            new_classes.append(classes[idx_low])
            new_classes.append(classes[idx_high])

    return dict(zip(new_classes, new_list))

################################################################################
# MAIN
################################################################################
def gen_splits(data_dir, dataset, sr, weight_list, type_):
    """Main split generator function

    Args:
        data_dir (str): The directory of the dataset samples, expects .npy files 
        dataset (str): Name of the dataset for file creation 
        sr (int): Sample rate of the data, or resample rate if being used for consistency
        weight_list (list): List of weights for the train/val/test splits 
        type_ (str): The type of split to generate, either stratified or shifted

    Raises:
        ValueError: If type of split is unrecognised, raise error
    """
    # Gets the names of the included classes
    contained_classes = os.listdir(data_dir)

    # Iterates over the full folder structure and counts how many wav files 
    total_files = 0
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.npy'):
                total_files += 1
    
    # A little visual to see how the processing is coming along
    progress_bar = tqdm(total=total_files)
    file_name = dataset +'.pkl'

    try:
        a_file = open(file_name, "rb")
        class_expected_values = pickle.load(a_file)

    except:
        class_expected_values = {}

        for seen_class in contained_classes:
                class_lengths = []
                working_dir = os.path.join(data_dir, seen_class)

                if len(os.listdir(working_dir)) < 2:
                    continue
                for fname in os.listdir(working_dir):
                    if fname.endswith('.npy'):
                        file_path = os.path.join(working_dir, fname)

                        length = load_and_length(file_path, sr)

                        class_lengths.append(length)

                        # Update the progress bar
                        progress_bar.update(1)

                
                #plot_dist(class_lengths, 300, seen_class, save=False)
                x, y = extract_kde(class_lengths)
                class_expected = greedy_expected_value(class_lengths, x, y)
                class_expected_values[seen_class] = class_expected

        file_name = dataset +'.pkl'
        a_file = open(file_name, "wb")
        pickle.dump(class_expected_values, a_file)
        a_file.close()

    sorted_dict = dict(sorted(class_expected_values.items(), key=lambda item: item[1]))

    if type_ == 'shifted':
        sorted_dict = dict(sorted(class_expected_values.items(), key=lambda item: item[1]))

    elif type_ == 'stratified':
        sorted_dict = stratified_sort(sorted_dict)
    else:
        raise ValueError('Please select a type of class-wise split distribution to generate')

    train, val, test = class_split(list(sorted_dict.keys()), weight_list)
    save_splits([train, val, test], dataset, type_)

    sns.histplot(list(sorted_dict.values()), bins=10, color='orange')
    plt.xlabel('Expected Value of Class (s)')
    plt.ylabel('Instances')
    plt.show()



################################################################################
# MAIN CALLS
################################################################################
SR = 16000
type_ = 'shifted' # shifted/stratified
WEIGHT_LIST = [0.7, 0.1, 0.2]
DATASET = 'BirdClef'
DATA_DIR = 'D:\Dataset Backups\Datasets\BirdSong\BirdClef2020\Sorted_npy'
load_path = True

# SR = 16000
# type_ = 'stratified' # shifted/stratified
# WEIGHT_LIST = [0.7, 0.1, 0.2]
# DATASET = 'Kaggle_18'
# DATA_DIR = 'C:/Users/user/Documents/Datasets/FSDKaggle18/Sorted_npy'
# #DATA_DIR = 'X:/Datasets/Kaggle AudioSet/Sorted_npy'
# load_path = True

# SR = 16000
# type_ = 'shifted' # shifted/stratified
# WEIGHT_LIST = [0.7, 0.1, 0.2]
# DATASET = 'VoxCeleb'
# #DATA_DIR = 'C:/Users/user/Documents/Datasets/FSDKaggle18/Sorted_npy'
# DATA_DIR = 'X:/Datasets/VoxCeleb1/Sorted_flatten_npy'
# load_path = True

gen_splits(DATA_DIR, DATASET, SR, WEIGHT_LIST, type_)
