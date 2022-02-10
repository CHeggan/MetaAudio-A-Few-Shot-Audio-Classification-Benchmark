"""
Script deals with:
    -> conversion of varibale length samples ot stacked spectrograms
    -> Load from .npy and save to .npy

The main function iterates over the so called 'old_dir'and creates a mirror
    directory in 'new_dir'.

We assume that we already have ful z-normalised samples in a two level class 
    storage system, i.e folders of classes within some main folder structure

We also assume that all samples are valid and none have to be
    removed. 
"""
###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import librosa
import numpy as np
from tqdm import tqdm

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
# TO SPECTROGRAM FUNCTION
###############################################################################
def single_to_spectrogram(data,  **spec_args):
    """Converts a raw audio signal into a spectogram and returns it

    Args:
        data (array): 
        spec_args (dictionary): Mel spectrogram variables for the conversion
    """
    data = np.nan_to_num(data)

    mel_spec = librosa.feature.melspectrogram(y=data,
                                            **spec_args)
    
    log_mel_spec = 20.0 / spec_args['power'] * np.log10(mel_spec + sys.float_info.epsilon)

    return log_mel_spec

###############################################################################
# STCAKED SPECTROGRAM FUNCTION
###############################################################################
def stacked_spec(og_file_path, new_dir, length, spec_params):
    """Creates a stacked fixed length spectrogram representation from a .npy 
        audio file

    Args:
        og_file_path (str): The exact path to the current file location
        new_dir (str): Exact path to teh new data save location
        length (int): The length in seconds that we create the representation for
        spec_params (dict): Dictionary of spectrogram paramaters
    """
    expected_size = spec_params['sr'] * length

    # Generates the new filename path
    file_name = og_file_path.split('\\')[-1]
    file_name = file_name.split('.')[0]
    new_path = os.path.join(new_dir, file_name)

    if os.path.isfile(new_path + '.npy'):
        return

    audio_sample = np.load(og_file_path)
    # Applies z-normalisation on a sample wise basis
    audio_sample = (audio_sample - np.mean(audio_sample)) / np.std(audio_sample)

    # Collects the raw sample splits
    raw_splits = []

    # If the sample is smaller than expected, we repeat it until we hit the
    #   mark and then trim back if needed
    if audio_sample.shape[0] < expected_size:
        # Calculates the number of repetitions needed of the sample
        multiply_up = int(np.ceil((expected_size) / audio_sample.shape[0]))
        sample = audio_sample.repeat((multiply_up, ))
        # Clips the new sample down as needed
        sample = sample[:expected_size]
        # Store or sample in a list to access later
        raw_splits.append(sample)

    # If the sample is longer than needed, we split it up into its slices
    elif audio_sample.shape[0] >= expected_size:
        starting_index = 0
        while starting_index < audio_sample.shape[0]:
            to_end = audio_sample.shape[0] - starting_index
            # If there more than a full snippet sample still available
            if to_end >= expected_size:
                split = audio_sample[starting_index:(starting_index + expected_size)]
                starting_index += expected_size
                raw_splits.append(split)

            # If we are at the end of our sample
            elif to_end < expected_size:
                # Calculates the number of repetitions needed of the sample
                multiply_up = int(np.ceil((expected_size) / to_end))
                split = audio_sample[starting_index:]
                # Repeats and clips the end sample as needed
                split = split.repeat((multiply_up, ))[:expected_size]
                starting_index = audio_sample.shape[0]
                raw_splits.append(split)

    mel_splits = []
    # Now need to convert the raw sample into melspectrogram
    # Need to convert to numpy and back again
    for raw in raw_splits:
        mel_spec = single_to_spectrogram(raw, **spec_params)
        mel_splits.append(mel_spec)

    x = np.stack(mel_splits)
    np.save(new_path, x)
    

###############################################################################
# MAIN FUNCTION
###############################################################################
def main(old_dir, new_dir, sample_length, spec_params):
    """The main sprctogram creation function for variable length datasets

    Args:
        old_dir (str): Path to current data directory
        new_dir (str): Path to create new mirrored directory at
        sample_length (int): The length for sample clipping and stacking in seconds
        spec_params (dict): Dictionary of spectrogram params
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
            if f.endswith('.npy'):
                total_files += 1
    
    # A little visual to see how the processing is coming along
    progress_bar = tqdm(total=total_files)

    for seen_class in contained_classes:
            working_dir = os.path.join(old_dir, seen_class)
            new_working_dir = os.path.join(new_dir, seen_class)

            try:
                os.mkdir(new_working_dir)
            except Exception:
                pass

            print(seen_class)

            if len(os.listdir(working_dir)) < 2:
                print(working_dir)
                sys.exit()

            for fname in os.listdir(working_dir):
                if fname.endswith('.npy'):
                    file_path = os.path.join(working_dir, fname)

                    stacked_spec(file_path, new_working_dir, sample_length, spec_params)
                    # Update the progress bar
                    progress_bar.update(1)

###############################################################################
# MAIN CALL
###############################################################################
# old_dir = 'D:/Dataset Backups/Datasets/BirdSong/BirdClef2020/Sorted_npy'
# new_dir = 'X:\Datasets\Bird Sounds\BirdClef 2020/Spec_5_seconds_npy_'

# sample_length = 5

# spec_params = {'sr': 16000,
#                 'n_mels':128,
#                 'n_fft':1024,
#                 'hop_length':512,
#                 'power':2.0}

# main(old_dir=old_dir,
#         new_dir=new_dir,
#         sample_length=sample_length,
#         spec_params=spec_params)

