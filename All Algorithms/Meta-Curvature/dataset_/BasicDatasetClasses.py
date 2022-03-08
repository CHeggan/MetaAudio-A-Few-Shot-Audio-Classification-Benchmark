"""
File contains the basic dataset class that is used for stats generation
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
###############################################################################
# GENERIC TRAINING DATASET
###############################################################################
"""
This class is only used to load a set of data and get its stats, i.e mean and std
    over all samples, either per channel or globally. Due to this, no normalisation
    or subset etc is added to teh options and processing.
"""
class BasicTrainingSet(Dataset):
    def __init__(self, path, classes):
        """Super basic dataset class, only used to iterate over the data and
            generate useful statistics

        Args:
            path (str): The exact path to the parent folder contsining the data 
            classes (list): All of the classes to be considered in the set
        """
        self.path = path
        self.classes = classes

        self.df = pd.DataFrame(self.get_subset())
        self.df = self.df.assign(id=self.df.index.values)

        # Grabs all the class names
        self.unique_characters = sorted(self.df['class_name'].unique())

        # Creates key:pair for class_name:numeric class_id
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}

        # Creates a class_id column in df using the class_name to class_id dict
        self.df = self.df.assign(
            class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Organises the sample ids and paths into two iterable arrays
        self.id_to_path = self.df.to_dict()['filepath']
        self.id_to_class_id = self.df.to_dict()['class_id']


    def __getitem__(self, item):
        """Grabs the requested data sample by id

        Args:
            item (int): The id of the sample in the dataset dataframe 

        Returns:
            Tensor, int: The uadio sample and its associated label
        """
        sample = np.load(self.id_to_path[item])
        sample = torch.from_numpy(sample)

        label = self.id_to_class_id[item]

        return sample ,label


    def __len__(self):
        return len(self.df)


    def num_classes(self):
        return len(self.df['class_name'].unique())


    def get_subset(self):
        """Iterates though the datasset path and grabs details of all relevant 
            class files that should be considered in part of the training split.

        Returns:
            list(dicts): An array of file instances, which each have their data stored in
                dictionary format
        """
        audio_samples = []

        for root, folders, files in os.walk(self.path):
            if len(files) == 0:
                continue

            class_name = root.split('\\')[-1]
            # Only look at if class is part of training split
            if class_name in self.classes:
                for f in files:
                    if f.endswith('.npy'):
                        audio_samples.append({
                            'class_name': class_name,
                            'filepath': os.path.join(root, f)
                            })
        return audio_samples

###############################################################################
# COPY OF MEL SPEC FUNCTION
###############################################################################
def mel_spec_function(x, sr, n_mels, n_fft, hop_length, power):
    mel_spec_array = librosa.feature.melspectrogram(y=x,
                                    sr=sr,
                                    n_mels=n_mels,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    power=power)

    log_mel_spec = 20.0 / power * np.log10(mel_spec_array + sys.float_info.epsilon)
    return log_mel_spec


###############################################################################
# STABLE NUM WORKERS DATALOADER
###############################################################################
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
