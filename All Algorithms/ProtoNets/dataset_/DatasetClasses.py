"""
Script contains the generalised dataset classes that are used for experiments,
    these include:
        -> General raw
        -> General spectrogram
    Each of these scripts is equiped with a variery of normalisation options

File also contains the basic base dataset classes that are used in the general 
    experiment framework. These datasets are also recycled 

Contains:
    -> Generic
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import torch
import numpy as np
import pandas as pd

from sklearn import preprocessing
from torch.utils.data import Dataset
from dataset_.dataset_stuff import per_sample_scale, nothing_func, given_stats_scale
###############################################################################
# SPECTROGRAM DATASET CLASS (GENERAL)
###############################################################################
class SpecDataset(Dataset):
    def __init__(self, data_path, classes, norm, stats_file_path):

        self.norm = norm
        self.classes = classes
        self.data_path = data_path

        self.norm_func = self.set_norm_func(norm, stats_file_path)

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

    def set_norm_func(self, norm, stats_file):
        """
        Sets the normlaisation fucntion to be used for parsed data samples. Options
            are None, l2, global, channel and per_sample. Any of these can be passed
            as strings to choose.

        :param norm: str
            The type of normlaisation to be used

        :return norm_func: function
            The normalisation function which can be used t parse over the data samples
        """

        if norm == 'l2':
            norm_func = preprocessing.normalize

        elif norm == 'None':
            norm_func = nothing_func

        elif norm == 'per_sample':
            norm_func = per_sample_scale

        elif norm == 'global':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale

        elif norm == 'channel':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale

        else:
            raise ValueError('Passes norm type unsupported')

        return norm_func

    def __getitem__(self, item):
        """
        This sub function deals with actually getting the data from source.

        :param item: int
            The index of the data sample to grab from specified subset

        :return sample: Tensor
            Proceessed data from the filepath found by iterable item
        :return label: int
            The numeric class catagory for the sample
        """
        sample = np.load(self.id_to_path[item])
        sample = torch.from_numpy(sample)

        # Deals with normalisation of various types
        if self.norm in ['global', 'channel']:
            sample = self.norm_func(sample, self.mu, self.sigma)

        else:
            sample = self.norm_func(sample)

        label = self.id_to_class_id[item]

        return sample ,label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def get_subset(self):
        """
        Function iterates through all the included classes/ files in the
            specificed subset and grabs metadate form them for storage in a df.
            Data grabbed is subset, class_name and filepath.

        :return audio_samples: [dict, dict, ...., dict]
            An array of file instances, which each have their data stored in
                dictionary format
        """
        audio_samples = []
        
        subset_len = 0
        for root, folders, files  in os.walk(self.data_path):
            subset_len += len([f for f in files if f.endswith('.npy')])

        for root, folders, files in os.walk(self.data_path):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            if class_name in self.classes:
                for f in files:
                    if f.endswith('.npy'):
                        audio_samples.append({
                            'class_name': class_name,
                            'filepath': os.path.join(root, f),
                            })
        return audio_samples

##############################################################################
# TRAINING VARIABLE LENGTH DATASET WITH SPECS
##############################################################################
class TrainingVariableSpecDataset(Dataset):
    def __init__(self, data_path, classes, norm, stats_file_path):

        self.norm = norm
        self.classes = classes
        self.data_path = data_path
        # Grabs the necessary normalisation function
        self.norm_func = self.set_norm_func(norm, stats_file_path)

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

    def set_norm_func(self, norm, stats_file):
        """
        Sets the normlaisation fucntion to be used for parsed data samples. Options
            are None, l2, global, channel and per_sample. Any of these can be passed
            as strings to choose.

        :param norm: str
            The type of normlaisation to be used

        :return norm_func: function
            The normalisation function which can be used t parse over the data samples
        """

        if norm == 'l2':
            norm_func = preprocessing.normalize

        elif norm == 'None':
            norm_func = nothing_func

        elif norm == 'per_sample':
            norm_func = per_sample_scale

        elif norm == 'global':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale

        elif norm == 'channel':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale

        else:
            raise ValueError('Passes norm type unsupported')

        return norm_func

    def __getitem__(self, item):
        """
        This sub function deals with actually getting the data from source. As
            we start with a vraible length sample, we have to first load as raw, 
            z-normalise and then create a fixed length sample.

        :param item: int
            The index of the data sample to grab from specified subset

        :return sample: Tensor
            Proceessed data from the filepath found by iterable item
        :return label: int
            The numeric class catagory for the sample
        """
        sample = np.load(self.id_to_path[item])

        # Convert to torch Tensor
        sample = torch.from_numpy(sample)

        idx = np.random.choice(sample.shape[0])
        sample = sample[idx]

        label = self.id_to_class_id[item]
        return sample, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def get_subset(self):
        """
        Function iterates through all the included classes/ files in the
            specificed subset and grabs metadate form them for storage in a df.
            Data grabbed is subset, class_name and filepath.

        :return audio_samples: [dict, dict, ...., dict]
            An array of file instances, which each have their data stored in
                dictionary format
        """
        audio_samples = []
        
        subset_len = 0
        for root, folders, files  in os.walk(self.data_path):
            subset_len += len([f for f in files if f.endswith('.npy')])

        for root, folders, files in os.walk(self.data_path):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            if class_name in self.classes:
                for f in files:
                    if f.endswith('.npy'):
                        audio_samples.append({
                            'class_name': class_name,
                            'filepath': os.path.join(root, f),
                            })
        return audio_samples

##############################################################################
# EVAL VARIABLE LENGTH DATASET
##############################################################################
class VariableSpecDataset(Dataset):
    def __init__(self, data_path, classes, norm, stats_file_path):

        self.norm = norm
        self.classes = classes
        self.data_path = data_path
        # Grabs the necessary normalisation function
        self.norm_func = self.set_norm_func(norm, stats_file_path)

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

    def set_norm_func(self, norm, stats_file):
        """
        Sets the normlaisation fucntion to be used for parsed data samples. Options
            are None, l2, global, channel and per_sample. Any of these can be passed
            as strings to choose.

        :param norm: str
            The type of normlaisation to be used

        :return norm_func: function
            The normalisation function which can be used t parse over the data samples
        """

        if norm == 'l2':
            norm_func = preprocessing.normalize

        elif norm == 'None':
            norm_func = nothing_func

        elif norm == 'per_sample':
            norm_func = per_sample_scale

        elif norm == 'global':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale

        elif norm == 'channel':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale

        else:
            raise ValueError('Passes norm type unsupported')

        return norm_func

    def __getitem__(self, item):
        """
        This sub function deals with actually getting the data from source. As
            we start with a vraible length sample, we have to first load as raw, 
            z-normalise and then create a fixed length sample.

        :param item: int
            The index of the data sample to grab from specified subset

        :return sample: Tensor
            Proceessed data from the filepath found by iterable item
        :return label: int
            The numeric class catagory for the sample
        """
        sample = np.load(self.id_to_path[item])

        # Convert to torch Tensor
        sample = torch.from_numpy(sample)

        label = self.id_to_class_id[item]
    
        return sample, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def get_subset(self):
        """
        Function iterates through all the included classes/ files in the
            specificed subset and grabs metadate form them for storage in a df.
            Data grabbed is subset, class_name and filepath.

        :return audio_samples: [dict, dict, ...., dict]
            An array of file instances, which each have their data stored in
                dictionary format
        """
        audio_samples = []
        
        subset_len = 0
        for root, folders, files  in os.walk(self.data_path):
            subset_len += len([f for f in files if f.endswith('.npy')])

        for root, folders, files in os.walk(self.data_path):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            if class_name in self.classes:
                for f in files:
                    if f.endswith('.npy'):
                        audio_samples.append({
                            'class_name': class_name,
                            'filepath': os.path.join(root, f),
                            })
        return audio_samples

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
