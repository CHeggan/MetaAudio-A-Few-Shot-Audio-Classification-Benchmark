"""
Main function for the pretraining of an arbirary model with a variety of datasets
"""

################################################################################
# IMPORTS
################################################################################
import os 
import sys
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from collections import Counter, OrderedDict
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from pretrain_fit import fit
from pretrain_utils import load_model
from model_selection import grab_dual_model
from dataset_.SetupClass import DatasetSetup
from dataset_.DatasetClasses import SpecDataset, FastDataLoader
from prep_batch_conventional import prep_batch, prep_trans_batch

###############################################################################
# MAIN
###############################################################################
if __name__ == '__main__':
    #########################
    # PRELIMINARIES
    #########################
    # Param import
    with open("experiment_params.yaml") as stream:
        params = yaml.safe_load(stream)

    # Setting of cuda device
    device = torch.device('cuda:' + str(params['base']['cuda']) if \
        torch.cuda.is_available() else 'cpu')
    print(device)


    #########################
    # DATASETS AND LOADERS
    #########################
    class_splits = None
    if params['data']['fixed']:
        class_splits = np.load(params['data']['fixed_path'], allow_pickle=True)
    
    # Option to use all available classes of dataset for pretraining
    if params['base']['use_all']:
        splits = [1, 0, 0]
    else:
        splits = [params['split']['train'], params['split']['val'], params['split']['test']]

    setup = DatasetSetup(dataset_name=params['data']['name'],
                    type=params['data']['type'],
                    data_path=params['data']['data_path'],
                    splits=splits,
                    norm=params['data']['norm'],
                    seed=params['base']['seed'],
                    num_workers=params['training']['num_workers'],
                    class_splits=class_splits)

    # Defines the datasets to be used
    dataset = SpecDataset

    full_set = dataset(data_path=params['data']['data_path'],
                    classes = setup.train,
                    norm=params['data']['norm'],
                    stats_file_path=setup.stats_file_path)

    id_to_class = full_set.id_to_class_id
    targets = list(id_to_class.values())

    # Generate train and test split indicies
    train_idx, valid_idx= train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        shuffle=True,
        stratify=targets)

    trainset = Subset(full_set, train_idx)
    validset = Subset(full_set, valid_idx)

    train_targets = np.array(targets)[train_idx]

    train_loader = FastDataLoader(trainset, batch_size=params['training']['batch_size'],
        num_workers=params['training']['num_workers'], shuffle=True)
    valid_loader = FastDataLoader(validset, batch_size=params['training']['batch_size'],
        num_workers=params['training']['num_workers'], shuffle=True)

    #########################
    # MODEL LOADING
    #########################
    with open("models/params/all_model_params.yaml") as stream:
        model_params = yaml.safe_load(stream)

    for name in params['models']:
        print(f'Using: {name}')
        model = grab_dual_model(name, model_params[name], out_dim=full_set.num_classes())
        model = model.to(device, dtype=torch.double)

        # load old model
        model = load_model('best_val_model__09_02__10_34.pt', model)

        #########################
        # LOSS & OPTIMISER
        #########################
        if params['training']['loss'] == 'weighted':
            # Counts up instances of classes in training set and generates weightings
            counter_dict = Counter(train_targets)
            od = OrderedDict(sorted(counter_dict.items()))
            class_weights = list(od.values())
            class_weights = torch.Tensor(class_weights).to(device, dtype=torch.double)

            # Creates a loss function which uses class weightings
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(device)

        elif params['training']['loss'] == 'normal':
            loss_fn = nn.CrossEntropyLoss().to(device)

        else:
            raise ValueError('Loss type not recognised')

        optimiser = optim.Adam(model.parameters(), lr=params['hyper']['initial_lr'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser,
                                            T_max=params['hyper']['T_max'],
                                            eta_min=params['hyper']['min_lr'])

        #########################
        # BATCH FUNCTIONS
        #########################
        if params['training']['batch_type'] == 'normal':
            prep_batch = prep_batch(device)
        elif params['training']['batch_type'] == 'trans':
            prep_batch = prep_trans_batch(device)
        else:
            raise ValueError('Batch type not recognised')

        #########################
        # MAIN FIT CALL
        #########################
        fit(model=model, 
            optimiser=optimiser, 
            scheduler=scheduler, 
            loss_fn=loss_fn, 
            dataloaders=[train_loader, valid_loader], 
            prep_batch=prep_batch, 
            params=params)
        

        

        







    
    