"""
File has a general pre-control file for dataset setup. These processes include:
    -> Generating class splits 
    -> Gathering training set stats to be used in global or channel normalisation
    -> Storing and searching for relevant norm stats to save computation effort

This dataset is created with the intent of being able to switch between use with
    either spectrograms or raw audio
"""

###############################################################################
# IMPORTS
###############################################################################
import os

from dataset_.dataset_stuff import class_split, check_for_stats, gen_training_stats, esc_class_split

###############################################################################
# SETUP CLASS
###############################################################################
class DatasetSetup:
    def __init__(self, dataset_name, type, data_path, splits, norm, seed, num_workers, class_splits):

        # Saves the data path to access later
        self.path = data_path
        # Checks to see if we need to generate stats
        if class_splits == None:
            # Unpacks all of the class subsets
            if dataset_name == 'ESC_strat':
                self.train, self.val, self.test = esc_class_split(data_path, splits)
            elif dataset_name in ['AudioSet', 'ESC', 'nsynth']:
                self.train, self.val, self.test = class_split(data_path, splits)
            else:
                raise ValueError('Check dataset name')

        # If we want to use a fixed set, we unpack here
        else:
            self.train, self.val, self.test = class_splits
            seed = 'fixed'

        if norm in ['global', 'channel']:
            # Defines the file name we are looking for, i.e 'global_AudioSet_spec_0.npy'
            split_string = [str(i) for i in splits]
            spl_str = "_".join(split_string)

            file_name = norm + '_' + dataset_name + '_' + type + '_' + str(seed) + \
                '__' + spl_str + '.npy'

            folder_name = 'dataset_/norm_stats'

            # Check for relevant stats files
            stats = check_for_stats(folder_name, file_name)
            save_path = os.path.join(folder_name, file_name)

            if stats[0] == None:
                stats = gen_training_stats(data_path=data_path, 
                                            training_classes=self.train, 
                                            norm=norm, 
                                            type=type, 
                                            save_path=save_path, 
                                            num_workers=num_workers)
            # Makes file_name a class variable so we can access it later 
            self.stats_file_path = save_path
            # Unpack the stats array
            self.mu, self.sigma = stats
        
        # If norm is in ['per_sample'] we dont need to gen these stats
        else:
            self.stats_file_path = None
            self.mu, self.sigma = 0, 1
