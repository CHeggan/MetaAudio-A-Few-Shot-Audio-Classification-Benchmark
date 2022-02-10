"""
File deals with taking in all of the unsorted train and test data and spits
    out a sorted directory of classes and their corresponding files
As this data will be getting used for meta-learning, all of it will go into
    one train folder, where test/ val splits can then be sampled from
The Kaggle18 set was created for a more traditional machine-learning approach in mind
    and so train and test share the exact same set of classes
"""

##############################################################################
# IMPORTS
##############################################################################
import os
import numpy as np
import pandas as pd
from shutil import copyfile

##############################################################################
# HELPER FUNCTIONS
##############################################################################
def create_class_folders(folder_list, sorted_train):
    """Creates the necessary data class folders

    Args:
        folder_list (list[str]): The list of folders to create 
        sorted_train (str): The main folder name for where all of these new
            folders go

    Returns:
        list[str]: A generated list of exact folder paths
    """
    folder_paths = []
    for i, folder in enumerate(folder_list):
        # creates teh necessary folder paths
        folder_path = os.path.join(main_dir, sorted_train, folder)
        folder_paths.append(folder_path)
        # Tries to create the folder selected, if it already exists, prints
        try:
            os.mkdir(folder_path)
        except Exception:
            print(f'Already Created: {folder_path}')

    return folder_paths

def grab_move_files(folders, folder_paths, og_data_path, df):
    """Sorts files form unstructured folder into class folders based on some 
        folder lists and file name connections

    Args:
        folders (list[str]): A list of all the folders/ class names 
        folder_paths (list[str]): A list of all of the new folder's exact paths 
        og_data_path (str): The path to the original data folder we are sorting
        df (dataframe): A dataframe that relates class name(label) to file name
    """
    for idx, folder_path in enumerate(folder_paths):
        label = folders[idx]
        sub_df = df[df['label'] == label]

        for j, fname in enumerate(sub_df['fname']):
            og_path = os.path.join(og_data_path, fname)
            new_path = os.path.join(folder_path, fname)
            copyfile(og_path, new_path)

##############################################################################
# MAIN FUNCTION
##############################################################################
def main(main_dir):
    """Main Kaggle18 sorting function. Reads in meta data required. Assumes being
        directed to the master folder

    Args:
        main_dir (str): Path to the Kaggle18 master folder
    """
    # Paths to the meta data files 
    meta_path_test = os.path.join(main_dir, 'FSDKaggle2018.meta', 'test_post_competition_scoring_clips.csv')
    meta_path_train = os.path.join(main_dir, 'FSDKaggle2018.meta', 'train_post_competition.csv')
    # Path to data
    test_folder = os.path.join(main_dir, 'FSDKaggle2018.audio_test')
    train_folder = os.path.join(main_dir, 'FSDKaggle2018.audio_train')

    # Sorted train, will go in main_dir
    sorted_train = 'Sorted'
    os.mkdir(os.path.join(main_dir, sorted_train))

    # Create meta dataframes for train and test
    test_df = pd.read_csv(meta_path_test)
    train_df = pd.read_csv(meta_path_train)

    # Joins the two sets into one large dataframe
    big_df = test_df.append(train_df)

    # Combines the folders form trian and test splits, this shouldnt actually
    #   need done as they share exact classes, but is done just incase there is 
    #   some difference
    folders = np.concatenate( (train_df['label'].unique(), test_df['label'].unique()) )
    folders = list(set(folders))
    
    # Creates all the necessary folders
    folder_paths = create_class_folders(folders, sorted_train)

    # Performs the sorting for the two sets, remember they go same folders 
    grab_move_files(folders, folder_paths, test_folder, test_df)
    grab_move_files(folders, folder_paths, train_folder, train_df)


##############################################################################
# MAIN CALL
##############################################################################

# # Main directory path
# main_dir = 'X:/Datasets/Kaggle AudioSet'

# main(main_dir)
