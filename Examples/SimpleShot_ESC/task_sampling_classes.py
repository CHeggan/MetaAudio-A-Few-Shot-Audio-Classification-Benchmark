"""
File contains the various task_sampling classes used in meta-learning:
    -> The NShotTaskSampler class inherits from sampler, has to have __iter__ and __len__
    -> Can specify specific type restrictions in class initialisations
"""

import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler, DataLoader


###############################################################################
# INTERSHUFFLE N-SHOT TASK SAMPLER
###############################################################################
class NShotTaskSampler(Sampler):
    """
    Task smapler that randomly samples n_way classes for every task. This is
        equivalent to intershuffle paradigm.

    :param episodes_per_epoch: int
        Num of batches of n-shot takss ot generate in one epoch
    :param n: int
        Number of classes sampled each task, dicates how many outputs model has
    :param k: int
        Number of support samples per class for classification tasks
    :param q: int
        Number of query samples for each class in tasks
    :param num_tasks: int
        Number of n-shot tasks to group into a single batch
    :param seed: int
        The seed to use in order to set np.random.seed(), important for
            reproducibility
    """
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n_way: int = None,
                 k_shot: int = None,
                 q_queries: int = None,
                 num_tasks: int = 1,
                 seed: int=0):

        super(NShotTaskSampler, self).__init__(dataset)
        self.dataset = dataset
        self.episodes_per_epoch= episodes_per_epoch
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_tasks = num_tasks
        np.random.seed(seed)

    def __len__(self):
        return self.episodes_per_epoch


    # One iteration of the task sampler when called
    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            # Itereates through the number of task to be created
            for task in range(self.num_tasks):
                # Ranodmly samples k different classes
                episode_classes = np.random.choice(
                    self.dataset.df['class_id'].unique(), size=self.n_way, replace=False)
                # Gets all samples with the randomly chosen class ids
                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                # Initialises a dictionary for the support set len sampled_classes
                # Only reason we track this in dict is to prevent support=query
                support_samps = {n: None for n in episode_classes}
                # Samples n examples of the each classes randomly chosen
                for n in episode_classes:
                    support = df[df['class_id']==n].sample(self.k_shot)
                    # Stores the sampled values in the support dictionary
                    # This means that kth element is keyed to a list of samples
                    support_samps[n] = support

                    # i, s are index and row of df respectively
                    for i, s in support.iterrows():
                        # Each sampled point is added to the batch
                        batch.append(s['id'])


                for n in episode_classes:
                    # Samples values from ktch class for the query set
                    # The & part is to check whether it is already in support
                    query = df[(df['class_id']==n) & (~df['id'].isin(support_samps[n]['id']))].sample(self.q_queries)

                    for i, q in query.iterrows():
                        batch.append(q['id'])

            # Yield pauses the function saving its states and later continues from there
            yield np.stack(batch)


###############################################################################
# STRATIFIED BATCH SAMPLER
###############################################################################
class StratifiedBatchSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 weight_list,
                 batch_size,
                 seed: int=0):

        super(StratifiedBatchSampler, self).__init__(dataset)
        self.dataset = dataset
        self.class_weights = weight_list
        self.batch_size = batch_size
        np.random.seed(seed)

    def __len__(self):
        return self.batch_size

    # One iteration of the task sampler when called
    def __iter__(self):
        batch = []

        available_classes = self.dataset.df['class_id'].unique()
        chosen_classes = np.random.choice(available_classes, size=self.batch_size, 
            p=self.class_weights)

        df = self.dataset.df[self.dataset.df['class_id'].isin(chosen_classes)]

        for idx, n in enumerate(chosen_classes):
            sample = df[df['class_id']==n].sample(1)
            for i, s in sample.iterrows():
                batch.append(s['id'])

        yield batch


        """
        for _ in range(self.episodes_per_epoch):
            batch = []

            # Itereates through the number of task to be created
            for task in range(self.num_tasks):
                # Ranodmly samples k different classes
                episode_classes = np.random.choice(
                    self.dataset.df['class_id'].unique(), size=self.n_way, replace=False)
                # Gets all samples with the randomly chosen class ids
                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                # Initialises a dictionary for the support set len sampled_classes
                # Only reason we track this in dict is to prevent support=query
                support_samps = {n: None for n in episode_classes}
                # Samples n examples of the each classes randomly chosen
                for n in episode_classes:
                    support = df[df['class_id']==n].sample(self.k_shot)
                    # Stores the sampled values in the support dictionary
                    # This means that kth element is keyed to a list of samples
                    support_samps[n] = support

                    # i, s are index and row of df respectively
                    for i, s in support.iterrows():
                        # Each sampled point is added to the batch
                        batch.append(s['id'])


                for n in episode_classes:
                    # Samples values from ktch class for the query set
                    # The & part is to check whether it is already in support
                    query = df[(df['class_id']==n) & (~df['id'].isin(support_samps[n]['id']))].sample(self.q_queries)

                    for i, q in query.iterrows():
                        batch.append(q['id'])

            # Yield pauses the function saving its states and later continues from there
            yield np.stack(batch)
        """