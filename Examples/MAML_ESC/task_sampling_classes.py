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
"""NON MUTUALLY EXCLUSIVE N-SHOT TASK SAMPLER"""
###############################################################################
class NMETaskSampler(Sampler):
    """
    Task sampler that splits all available classes into sets of len(n) and uses
        these splits for classes in a given task. Doing this fixes class labels
        when passed to theh model as well as fixing task types. This is an example
        of non-mutually exclusive task selection

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

        super(NMETaskSampler, self).__init__(dataset)
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_tasks = num_tasks
        np.random.seed(seed)
        # The sets of classes that enforce non mutual exclusivity
        self.class_sets = self.nonMutual(n_way)

    def nonMutual(self, n_way):
        # Grabs all the available class IDs
        class_ids = self.dataset.df['class_id'].unique()
        np.random.shuffle(class_ids)
        possible_tasks = int(np.floor(len(class_ids)/n_way))
        class_sets = [class_ids[(i*n_way):(i*n_way) + n_way] for i in range(possible_tasks)]
        return class_sets

    def __len__(self):
        return self.episodes_per_epoch

    # One iteration of the task sampler when called
    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            # Itereates through the number of task to be created
            for task in range(self.num_tasks):
                task_index = np.random.randint(0, len(self.class_sets), size=1)
                episode_classes = self.class_sets[task_index[0]]
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
"""OG INTERSHUFFLE N-SHOT TASK SAMPLER"""
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

        self.clean_dataset(dataset)

        np.random.seed(seed)

    def __len__(self):
        return self.episodes_per_epoch

    def clean_dataset(self, dataset):
        """Cleans the datasets temporarily for classes that do not have a
            sufficient number of samples for the porblem at hand

        Args:
            dataset (torch dataset): A dataset object that has a df of samples
                that can be iterated over
        """
        for n in dataset.df['class_id'].unique():
            sub_df = dataset.df[dataset.df['class_id'] == n]
            if sub_df.shape[0] < (self.k_shot + self.q_queries):
                dataset.df.drop(dataset.df.index[dataset.df['class_id'] == n], inplace=True)


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
