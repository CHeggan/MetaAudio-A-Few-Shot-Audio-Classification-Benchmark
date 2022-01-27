"""
Script deals with the cascasded initial pre-processing of the NSYNTH dataset.
    This contains the following:
        -> Sorting files into class folders based on meta data
        -> Converting raw wav files into npy arrays
        -> Taking raw audio saved as npy and converting into mel spectrograms

The idea is for this script to be ran once when dataset is originally downloaded
    and then if any corrections or changes have to made to the data, i.e a new
    number of n_mels in the spectrogram, then that specific script can be re run.

At the end of the script, should have the following directories of data:
    -> Original unsorted data (should )
    -> Sorted raw audio into class folders
    -> Sorted, per sample normalised and .npy converted audio - this should be
        the go to audio folder from now on due to faster loading and normalisation
    -> Spectrogram directory with no additional normalisaton, either per channel or global
"""


###############################################################################
# IMPORTS AND PARAMATERS
###############################################################################
import os

# Grabs all of the relevent main fucntions from associated scripts
from to_spec import main as to_spec_main
from to_np_and_norm import main as np_norm_main
from folder_sort_NSYNTH import main as folder_sort_main

#MAIN_DIR = ''
#FINAL_SPEC_PATH = ''

# Main NSYNTH folder path
MAIN_DIR = 'X:/Datasets/NSynth'
#MAIN_DIR = 'C:/Users/user/Documents/Datasets/NSYNTH'
# Path wated for final spectrogram data
#FINAL_SPEC_PATH = 'C:/Users/user/Documents/Datasets/NSYNTH/NSYNTH_spec'
FINAL_SPEC_PATH = 'X:/Datasets/NSynth/NSYNTH_spec'

NORM = True
SAMPLE_LENGTH = 4
MEL_SPEC_PARAMS = {'sr': 16000,
                'n_mels':128,
                'n_fft':1024,
                'hop_length':512,
                'power':2.0}

###############################################################################
# STACK CALL
###############################################################################
if __name__ == '__main__':
    # Sorts the mixed up audio files into classes
    sorted_path = folder_sort_main(main_dir=MAIN_DIR)
    sorted_path = os.path.join(MAIN_DIR, sorted_path)

    # Generates a new path for the sorted classes in npy format
    sorted_npy_path = os.path.join(MAIN_DIR, sorted_path + '_npy')
    np_norm_main(old_dir=sorted_path, new_dir=sorted_npy_path, sr=MEL_SPEC_PARAMS['sr'], norm=NORM)

    # Carries out the spectrogram dataste creation
    to_spec_main(old_dir=sorted_npy_path,
                    new_dir=FINAL_SPEC_PATH,
                    sample_length=SAMPLE_LENGTH,
                    spec_params=MEL_SPEC_PARAMS)
