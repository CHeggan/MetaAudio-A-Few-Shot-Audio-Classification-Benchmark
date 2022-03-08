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

from dataset_.SetupClass import DatasetSetup
from Pre.pretrain_fit import fit
from Pre.pretrain_utils import load_model, set_seed
from Pre.model_selection import grab_model
from prep_batch_functions import prep_batch_fixed, prep_var_eval, prep_batch_conventional
from dataset_.DatasetClasses import NormDataset, FastDataLoader, TrainingVariableDataset

###############################################################################
# DATALOADER COLLATE FUNCTION
###############################################################################
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

###############################################################################
# MAIN
###############################################################################
def main(main_params, model_name):
    #########################
    # PRELIMINARIES
    #########################
    # Param import
    with open("Pre/experiment_params.yaml") as stream:
        params = yaml.safe_load(stream)

    # Setting of cuda device
    device = torch.device('cuda:' + str(main_params['base']['cuda']) if \
        torch.cuda.is_available() else 'cpu')

    # Fully sets the seed of the run
    set_seed(main_params['base']['seed'])

    params['base']['seed'] = main_params['base']['seed']
    params['base']['task_type'] = 'PRE_' + main_params['base']['task_type']

    #########################
    # DATASETS AND LOADERS
    #########################
    class_splits = None
    if main_params['data']['fixed']:
        class_splits = np.load(main_params['data']['fixed_path'], allow_pickle=True)
    
    # Option to use all available classes of dataset for pretraining
    if params['base']['use_all']:
        splits = [1, 0, 0]
    else:
        splits = [main_params['split']['train'], main_params['split']['val'], main_params['split']['test']]

    # Runs the setup class for the dataset
    setup = DatasetSetup(params=main_params,
                        splits=splits,
                        seed=main_params['base']['seed'],
                        class_splits=class_splits)

    # Defines the datasets to be used
    if main_params['data']['variable']:
        train_dataset = TrainingVariableDataset
        val_dataset = NormDataset
        val_coll = my_collate
    else:
        train_dataset = NormDataset
        val_dataset = NormDataset
        val_coll = None

    # define the full set so we can grab indices
    full_set = train_dataset(data_path=main_params['data']['data_path'],
                    classes = setup.train,
                    norm=main_params['data']['norm'],
                    stats_file_path=setup.stats_file_path)

    valid_set = val_dataset(data_path=main_params['data']['data_path'],
                    classes = setup.train,
                    norm=main_params['data']['norm'],
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
    validset = Subset(valid_set, valid_idx)

    train_targets = np.array(targets)[train_idx]

    train_loader = FastDataLoader(trainset, batch_size=params['training']['batch_size'],
        num_workers=params['training']['num_workers'], shuffle=True)
    valid_loader = FastDataLoader(validset, batch_size=params['training']['batch_size'],
        num_workers=params['training']['num_workers'], shuffle=True, collate_fn=val_coll)

    #########################
    # MODEL LOADING
    #########################
    with open("Pre/models/params/all_model_params.yaml") as stream:
        model_params = yaml.safe_load(stream)

    print(model_params)


    print(f'Using: {model_name}')

    model = grab_model(model_name, model_params[model_name], out_dim=full_set.num_classes())
    model = model.to(device, dtype=torch.double)

    # Load semi-trained model
    # model = load_model('Pre/best_val_model__08_02__12_33.pt', model)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser,
                                    mode='min',
                                    patience=params['hyper']['patience'],
                                    factor=params['hyper']['factor'],
                                    min_lr=params['hyper']['min_lr'])

    #########################
    # BATCH FUNCTIONS
    #########################
    if main_params['data']['variable']:
        batch_func = prep_batch_conventional
    elif main_params['data']['variable'] == False:
        batch_func = prep_batch_conventional
    else:
        raise ValueError('What kind of data is it then?')

    prep_batch = batch_func(device, main_params['training']['trans_batch'])


    #########################
    # MAIN FIT CALL
    #########################
    best_model_path, model, best_val = fit(model=model, 
            optimiser=optimiser, 
            scheduler=scheduler, 
            loss_fn=loss_fn, 
            dataloaders=[train_loader, valid_loader], 
            prep_batch=prep_batch, 
            params=params,
            device=device,
            variable=main_params['data']['variable'])
    
    return best_model_path, model, best_val
    

        
        







    
    