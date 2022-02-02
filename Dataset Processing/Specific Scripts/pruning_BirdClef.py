"""
Expeirments with BirdClef require significantly more computational hardware than
    the other datasets considered. This is primarily due to the incredibly long 
    samples contained within some of the classes.

In order to make future exprimentation more accessible, we propose pruning the
    dataset with respect to length of samples. 

To make applying this code more simple, we have it work over an already existing 
    version of the BirdClef dataset, applying the prune as it iterates through.
    For added speedup, we allow an option for a csv save of files to be removed etc
   
"""
###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

###############################################################################
# SAMPLE LENGTH CALC
###############################################################################
def load_and_length(data_path, sr):
    data = np.load(data_path)
    length = float(data.shape[0]/sr)
    return length 

###############################################################################
# MAIN
###############################################################################
def main(main_dir, time_thresh_s, class_thresh,  SR, remove, csv_path=None):

    try:
        bad_files = pd.read_csv(csv_path)
        already_parsed = True

    except:
        bad_files = pd.DataFrame(columns=['class', 'file_name'])
        already_parsed = False

    if already_parsed:
        for idx in range(bad_files.shape[0]):
            class_name = bad_files['class'].iloc[idx]
            file = bad_files['file_name'].iloc[idx]
            file_path = os.path.join(main_dir, class_name, file)

            if remove:
                try:
                    os.remove(file_path)
                except:
                    pass
            
    else:
        # Gets the names of the included classes
        contained_classes = os.listdir(main_dir)
        # Iterates over the full folder structure and counts how many wav files 
        total_files = 0
        for root, _, files in os.walk(main_dir):
            for f in files:
                if f.endswith('.npy'):
                    total_files += 1
        
        # A little visual to see how the processing is coming along
        progress_bar = tqdm(total=total_files)

        num_removed = 0
        for seen_class in contained_classes:
                working_dir = os.path.join(main_dir, seen_class)

                for fname in os.listdir(working_dir):
                    if fname.endswith('.npy'):
                        file_path = os.path.join(working_dir, fname)
                        sample_length = load_and_length(file_path, SR)

                        # If sample length is too long, we dont want the sample
                        if sample_length > time_thresh_s:
                            new_point = {'class':seen_class, 'file_name':fname}
                            bad_files = bad_files.append(new_point, ignore_index=True)

                            if remove:
                                os.remove(file_path)
                            num_removed += 1

                        # Update the progress bar
                        progress_bar.update(1)

        # Can only do folder cleaning when we have already removed samples too long
        if remove:
            # We perform a class sweep after all of this, where we want a min num samples
            num_valid = 0
            for seen_class in contained_classes:
                class_path = os.path.join(main_dir, seen_class)
                samples = os.listdir(class_path)

                if len(samples) < class_thresh:
                    # need to remove all the files individually first before del folder
                    for file in samples:

                        file_path = os.path.join(main_dir, seen_class, file)

                        new_point = {'class':seen_class, 'file_name':file}
                        bad_files = bad_files.append(new_point, ignore_index=True)

                        num_removed += 1
                        os.remove(file_path)

                    # del folder
                    os.rmdir(os.path.join(main_dir, seen_class))
                        
                else:
                    num_valid += 1

            print('Number of classes Remaining:', num_valid)

        # print('Num Samples Removed:', num_removed)
        print(bad_files)
        bad_files.to_csv('remove_files.csv', index=False)

###############################################################################
# MAIN CALL
###############################################################################
main_path = 'D:/Dataset Backups/Datasets/BirdSong/BirdClef2020/Sorted_npy_copy'
csv_path = 'remove_files.csv'

main(main_dir=main_path,
    time_thresh_s=180, # Maximum time in secinds
    class_thresh=50, # Min num samples per class
    SR=16000, #Sample rate
    remove=True) #Should we remove the samples yet
