"""
Script deals with conversion form raw audio signal to mel-spectrogram. The 
    converson is performed over a singel folder which is assumed to contain
    multiple class wise folders with samples stored inside with extension .npy.

The files loaded should be .npy as this is a significant speed up. wav to npy
    converson is dealt with in another script. 

When doing this conversion we want to create a new folder with the spectrograms
    and not overwrite the sprted .npy raw audio signals. 

We dont worry about raw audio normalisation as it is handles in the 
    "to_np_and_norm.py" script. Spectrogram normalisation is a more complex and 
    split dpeendant task and so it is done during experiments and not as 
    pre-processing
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import librosa
import numpy as np
from tqdm import tqdm

###############################################################################
# SPECTRGRAM FUNCTION
###############################################################################
def to_spectrogram(og_file_path, new_dir, length, **spec_args):
    """Converts a raw audio signal into a spectogram and saves to a new path

    Args:
        og_file_path (str): 
        new_dir (str): The new class-wise working directory for the spectrogram data
        length (int, float or None): The expected length(in seconds) of the sample,
            should be used when dataset should have standardised length samples
    """
    audio_data = np.load(og_file_path)

    if np.std(audio_data) == 0.0:
        print(f'File has 0 std: {og_file_path}')
        return
    
     if audio_data.shape[0] < 160000:
        return

    audio_sum = np.sum(audio_data)
    if np.isnan(audio_sum):
        return

    #If length is None, means that samples are multi-length, i.e FSD2018, so
    #   checking length isnt reasonable
    if length != None:
        if audio_data.shape[0] != spec_args['sr']*length:
            print(f'Unsutable length: {audio_data.shape[0]}:: {og_file_path}')
            return

    mel_spec = librosa.feature.melspectrogram(y=audio_data,
                                            **spec_args)
    
    log_mel_spec = 20.0 / spec_args['power'] * np.log10(mel_spec + sys.float_info.epsilon)

    # Generates teh new filename path
    file_name = og_file_path.split('\\')[-1]
    file_name = file_name.split('.')[0]
    new_path = os.path.join(new_dir, file_name)

    np.save(new_path, log_mel_spec)


###############################################################################
# MAIN FUNCTION
###############################################################################
def main(old_dir, new_dir, sample_length, spec_params):
    """ Mian spec conversion function. Deals with actual iteration thorugh the dataset

    Args:
        old_dir (str): [The old data directory path - exact
        new_dir (str): The new data directory path - exact
        sample_length (int): The sample length to read in the samples at
        spec_params (dict): dictionary of spectrogram parameters
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
                print(f'Already Created: {new_working_dir}')

            for fname in os.listdir(working_dir):
                if fname.endswith('.npy'):
                    file_path = os.path.join(working_dir, fname)

                    to_spectrogram(file_path, new_working_dir, sample_length, **spec_params)
                    # Update the progress bar
                    progress_bar.update(1)

###############################################################################
# MAIN CALL
###############################################################################
"""
old_dir = 'X:\Datasets\ESC-50-master\Sorted_npy'
new_dir = 'X:\Datasets\ESC-50-master\ESC_spec'
sample_length = 5

spec_params = {'sr': 16000,
                'n_mels':128,
                'n_fft':1024,
                'hop_length':512,
                'power':2.0}

main(old_dir=old_dir,
        new_dir=new_dir,
        sample_length=sample_length,
        spec_params=spec_params)
"""
