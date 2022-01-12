"""
Goal of script is to visualise some generated split. Output is 4 sample length
    histograms with overlayed density curves. 
"""

################################################################################
# IMPORTS
################################################################################
import os 
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

################################################################################
# HELPER FUNCTIONS
################################################################################
def load_and_length(data_path, sr):
    data = np.load(data_path)
    length = float(data.shape[0]/sr)
    return length 


def create_dist(length_values, title, save=False):
    sns.histplot(length_values, bins=int(max(length_values)), kde=True, color='green')
    plt.xlabel('Length in Seconds')
    plt.xlim(0, int(max(length_values)) )
    plt.ylabel('Instances')
    plt.title(title)
    if save:
        plt.savefig(title + '.svg')
    plt.show()


def extract_kde(length_values):
    my_kde = sns.kdeplot(length_values, gridsize=200)
    line = my_kde.lines[0]
    x, y = line.get_data()
    plt.plot(x, y)
    plt.show()


def calc_expected_avg(length_values, split):
    my_kde = sns.kdeplot(length_values, gridsize=200)
    line = my_kde.lines[0]
    x, y = line.get_data()
    plt.clf()
    x, y = np.array(x), np.array(y)
    ex_avg = np.sum(np.array(x) * np.array(y)) / sum(y)
    print(split, '::', round(ex_avg, 3))


################################################################################
# MAIN
################################################################################
def main(data_dir, use_splits, SPLITS_PATH, SR, dataset):
    """Main function for visualising the dataset splits

    Args:
        data_dir (str): The path to the dataset directory
        use_splits (boolean): Whether or not to use a set split 
        SPLITS_PATH (str): Path to the split to use, expects .npy file
        SR (int): Sample/resample rate of the data 
        dataset (str): Name of the dataset, for file saving etc
    """
    if use_splits:
        train, val, test = np.load(SPLITS_PATH, allow_pickle=True)

    print('Train Classes: ', len(train))
    print('Val Classes: ', len(val))
    print('Test Classes: ', len(test))

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

    train_lengths = []
    val_lengths = []
    test_lengths = []

    all_lengths = []

    # Iterate through all classes and sample, splitting up into the train/val/test splits
    for seen_class in contained_classes:
            working_dir = os.path.join(data_dir, seen_class)

            for fname in os.listdir(working_dir):
                if fname.endswith('.npy'):
                    file_path = os.path.join(working_dir, fname)

                    length = load_and_length(file_path, SR)

                    if seen_class in train:
                        train_lengths.append(length)
                    if seen_class in val:
                        val_lengths.append(length)
                    if seen_class in test:
                        test_lengths.append(length)

                    all_lengths.append(length)

                    # Update the progress bar
                    progress_bar.update(1)

    print('\n')
    # Save files trigger
    save = True
    create_dist(train_lengths, 'TRAIN ' + dataset, save)
    create_dist(val_lengths, 'VAL ' + dataset, save)
    create_dist(test_lengths, 'TEST '+ dataset, save)
    create_dist(all_lengths, 'ALL '+ dataset, save)

    # Prints out expected values for each split based on all data
    calc_expected_avg(train_lengths, 'TRAIN')
    calc_expected_avg(val_lengths, 'VAL')
    calc_expected_avg(test_lengths, 'TEST')

################################################################################
# MAIN CALLS
################################################################################

# SR = 16000
# DATASET = 'Kaggle18_SHIFT'
# use_splits = True
# #SPLITS_PATH = 'splits/Kaggle18_norm_split.npy'
# SPLITS_PATH = 'splits/Kaggle18_shifted_split.npy'
# #SPLITS_PATH = 'splits/VoxCeleb_shifted_split.npy'
# #DATA_DIR = 'X:/Datasets/VoxCeleb1/Sorted_flatten_npy'
# DATA_DIR = 'C:/Users/user/Documents/Datasets/FSDKaggle18/Sorted_npy'
# #DATA_DIR = 'X:/Datasets/Kaggle AudioSet/Sorted_npy'
# main(DATA_DIR, use_splits, SPLITS_PATH, SR, DATASET)

SR = 16000
DATASET = 'BirdClef_'
use_splits = True
SPLITS_PATH = 'splits/BirdClef_shifted_split.npy'
DATA_DIR = 'D:\Dataset Backups\Datasets\BirdSong\BirdClef2020\Sorted_npy'
main(DATA_DIR, use_splits, SPLITS_PATH, SR, DATASET)