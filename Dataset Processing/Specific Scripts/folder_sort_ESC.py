"""
File deals with taking in all of the unsorted train and test data and spits
    out a sorted directory of classes and their corresponding files
As this data will be getting used for meta-learning, all of it will go into
    one train folder, where test/ val splits can then be sampled from
This set was created for a more traditional machine-learning approach in mind
    and so train and test share the exact same set of classes

This script specifically targets ESC-50 dataset and assumes that the mats data
    file meta.csv is in the main directory (as it is when downloaded form source)
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
def create_class_folders(main_dir, folder_list, sorted_train):
    """Creates the necessary data class folders

    Args:
        main_dir (str): The path of teh main working dorectory of ESC-50
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
        sub_df = df[df['category'] == label]

        for j, fname in enumerate(sub_df['filename']):
            og_path = os.path.join(og_data_path, fname)
            new_path = os.path.join(folder_path, fname)
            copyfile(og_path, new_path)

##############################################################################
# MAIN FUNCTION
##############################################################################
def main(main_dir):
    """Grabs and loads of the relevent meta files and folder paths before moving
        on to running the necessary helper and conversion fucntions in required
        order

    Args:
        main_dir (str): The raw path to the main dataset directory where all 
            folders and files should be made/read from
    
    Return:
        (str): The path to the sorted folders of audio
    """
    # Paths to the meta data files 
    meta_path = os.path.join(main_dir, 'meta', 'esc50.csv')

    # Path to data
    data_folder = os.path.join(main_dir, 'audio')

    # Sorted train, will go in main_dir
    sorted_train = 'Sorted'
    try:
        os.mkdir( os.path.join(main_dir, sorted_train) )

    except Exception:
        print('Sorted folder already exists')

    # Create meta dataframes for train and test
    data_df = pd.read_csv(meta_path)

    # Grabs the unique class names
    folders = data_df['category'].unique()
    
    # Creates all the necessary folders
    folder_paths = create_class_folders(main_dir, folders, sorted_train)

    # Performs the sorting for the two sets, remeber they go same folders 
    grab_move_files(folders, folder_paths, data_folder, data_df)
    return sorted_train


##############################################################################
# MAIN CALL
##############################################################################

# Main directory path
# Desktop
#main_dir = 'X:\Datasets\ESC-50-master'

#main(main_dir)
