"""
File and its functions deals with the sorting of the raw NSYNTH data containing 
    folders into their class specific folders. Have an additional functio here 
    to deal with proper manipulation of the meta data upon import.

Like the other inclded scripts here for the NSYNTH set, MAIN_DIR is expected to
    point to a parent directory with the relevant train/val/test .tar.gz files
    unpacked such that MAIN_DIR contains 'nsynth-test', 'nsynth-train' and 
    'nsynth-valid'
"""

##############################################################################
# IMPORTS
##############################################################################
import os
import json
import pandas as pd
from shutil import copyfile

from tqdm import tqdm

##############################################################################
# META DATA EXTRACTION
##############################################################################
def get_meta_data_nsynth(main_dir, sub_folders):
    """Function grabs the examples.json files from all sub nsynth folders,
        converts them and stores them as a large meta dataframe. Also saves the
        dataframe for later use.

    Args:
        main_dir (str): The exact path to the main nsynth data directory where
            teh subfolders are contained 
        sub_folders (list[str, ..., str]): A list of the main data subfolders
            which we need to consider when looking at meta data, train/val/test

    Returns:
        DataFrame: The large concatenated meta dataframe
    """
    # A collection of the extracted dataframes =
    extracted_dfs = []
    # Iterates over the subfolders, converting and storing the info
    for folder in sub_folders:
        audio_path = os.path.join(folder, 'audio')
        path = os.path.join(main_dir, folder, 'examples.json')

        with open(path) as f:
            data = json.load(f)
        # Flattens the weird dictonary structure for file names, making access easier
        new_data = []
        for key in data.keys(): 
            temp_line = data[key]
            temp_line['file_name'] = key+'.wav'
            # We grab and store the relative path from main directory while here
            temp_line['path_from_main'] = os.path.join(audio_path, key +'.wav')
            new_data.append(temp_line)

        extracted_dfs.append( pd.DataFrame(new_data) )
    
    # Concatenates all of the extracted dataframes
    meta_df = pd.concat(extracted_dfs)
    # Save the dataframe for potential later use in stratification etc
    meta_df.to_csv( os.path.join(main_dir, 'meta_df.csv') )
    return meta_df
        

##############################################################################
# HELPER FUNCTIONS
##############################################################################
def create_class_folders(main_dir, folder_list, sorted_train):
    """Creates the necessary data class folders

    Args:
        main_dir (str): The path of the main working directoty
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

def grab_move_files_nsynth(main_dir, folders, folder_paths, df):
    """Sorts files from unstructured folder into class folders based on some 
        folder lists and file name connections

    Args:
        folders (list[str]): A list of all the folders/ class names 
        folder_paths (list[str]): A list of all of the new folder's exact paths 
        df (dataframe): A dataframe that relates class name(label) to file name
    """
    # Set up progress bar to help track
    progress_bar = tqdm(total=len(df), desc='File Sorting')

    for idx, folder_path in enumerate(folder_paths):
        label = folders[idx]
        sub_df = df[df['instrument_str'] == label]

        for j, fname in enumerate(sub_df['file_name']):
            og_path = os.path.join(main_dir, sub_df.iloc[j]['path_from_main'])
            new_path = os.path.join(folder_path, fname)
            copyfile(og_path, new_path)
            progress_bar.update(1)

##############################################################################
# MAIN FUNCTION AND EXAMPLE CALL
##############################################################################
def main(main_dir):
    """Sorts the raw nsynth data into class folders based on instrument used

    Args:
        main_dir (str): The main nsynth data directory

    Returns:
        str: The folder name of the sorted raw samples
    """
    # The sub folders of data for nsynth
    sub_folders = ['nsynth-train', 'nsynth-test', 'nsynth-valid']
    # Grabs and saves relevant meta data
    meta_df = get_meta_data_nsynth(main_dir, sub_folders)

    # Get the class/folder names as strings
    instrument_strings = meta_df['instrument_str'].unique()

    # Sorted train, will go in main_dir
    sorted_train = 'Sorted_nsynth'
    try:
        os.mkdir( os.path.join(main_dir, sorted_train) )

    except Exception:
        print('Sorted folder already exists')

    # Creates the folders for sorting
    folder_paths = create_class_folders(main_dir, instrument_strings, sorted_train)

    # Moves and sorts teh nsynth data
    grab_move_files_nsynth(main_dir, instrument_strings, folder_paths, meta_df)
    return sorted_train

"""
MAIN_DIR = 'C:/Users/user/Documents/Datasets/NSYNTH'
main_folder_sort(MAIN_DIR)
"""

