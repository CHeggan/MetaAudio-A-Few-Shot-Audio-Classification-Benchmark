"""
Considerations include:
    -> Inverse class weighted loss function
    -> data loading 
    -> model loading
"""

##############################################################################
# IMPORTS
##############################################################################
import sys
import yaml

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import Counter, OrderedDict

from simpleshot_fit import fit_ss
from model_selection import grab_dual_model
from ss_steps import ss_eval_step_fixed, ss_eval_step_var

from dataset_.SetupClass import DatasetSetup
from task_sampling_classes import NShotTaskSampler, StratifiedBatchSampler
from dataset_.DatasetClasses import NormDataset, FastDataLoader, TrainingVariableDataset

# VoxCeleb specific files
from dataset_.VC_dataset import Vox_Dataset, TrainVox_Dataset
from dataset_.VC_task_sampling import Vox_NShotTaskSampler


###############################################################################
# DATALOADER COLLATE FUNCTION
###############################################################################
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

##############################################################################
# SINGLESHOT MAIN 
##############################################################################    

def single_run_main(params, model_name, device, prep_batch_fns, seed):
    # Loads in model params
    with open("models/params/all_model_params.yaml") as stream:
        model_params = yaml.safe_load(stream)

    #########################
    # DATASETS AND LOADERS
    #########################
    class_splits = None
    if params['data']['fixed']:
        class_splits = np.load(params['data']['fixed_path'], allow_pickle=True)
    
    splits = [params['split']['train'], params['split']['val'], params['split']['test']]

    # Runs the setup class for the dataset
    setup = DatasetSetup(params=params,
                        splits=splits,
                        seed=seed,
                        class_splits=class_splits)

   # Defines the datasets to be used
    if params['data']['variable']:
        train_coll = None
        val_coll = my_collate
        eval_step = ss_eval_step_var

        if params['data']['name'] == 'VoxCeleb':
            train_dataset = TrainVox_Dataset
            val_dataset = Vox_Dataset

            # If we want to use awareness at validation
            if params['data']['val_aware_sampling']:
                val_task_sampler = Vox_NShotTaskSampler
            else:
                val_task_sampler = NShotTaskSampler
        
        else:
            train_dataset = TrainingVariableDataset
            val_dataset = NormDataset
            val_task_sampler = NShotTaskSampler


    else:
        train_dataset = NormDataset
        val_dataset = NormDataset
        train_coll = None
        val_coll = None
        eval_step = ss_eval_step_fixed
        train_task_sampler = NShotTaskSampler
        val_task_sampler = NShotTaskSampler

    # Defines the datasets to be used
    train_set = train_dataset(data_path=params['data']['data_path'],
                            classes = setup.train,
                            norm=params['data']['norm'],
                            stats_file_path=setup.stats_file_path)

    # we define a flattened version of the train set so we include all samples in var length
    all_sample_train_set = val_dataset(data_path=params['data']['data_path'],
                        classes = setup.train,
                        norm=params['data']['norm'],
                        stats_file_path=setup.stats_file_path)

    validation = val_dataset(data_path=params['data']['data_path'],
                        classes = setup.val,
                        norm=params['data']['norm'],
                        stats_file_path=setup.stats_file_path)
    evaluation = val_dataset(data_path=params['data']['data_path'],
                        classes = setup.test,
                        norm=params['data']['norm'],
                        stats_file_path=setup.stats_file_path)

    # Gets the number of train classes 
    num_train_classes = train_set.num_classes()

    """
    # Set up the dataloading objects, we train more conventionally for D_base
    train_loader = FastDataLoader(train_set, num_workers=params['training']['num_workers'], 
        batch_sampler=StratifiedBatchSampler(
            dataset=train_set,
            weight_list=None,
            batch_size=params['training']['batch_size'],
            seed=seed)
        )
    """

    # Flattened train dataset for var length, make no difference for fixed
    train_loader = FastDataLoader(train_set, num_workers=params['training']['num_workers'], 
        batch_size=params['training']['batch_size'], collate_fn=train_coll, shuffle=True)

    id_to_class = train_set.id_to_class_id
    train_targets = np.array(list(id_to_class.values()))

    # Flattened train dataset for var length, make no difference for fixed
    flat_trainloader = FastDataLoader(all_sample_train_set, num_workers=params['training']['num_workers'], 
        batch_size= 1000, collate_fn=val_coll, shuffle=True)

    valTaskloader = FastDataLoader(
        validation, batch_sampler= val_task_sampler(dataset=validation,
                                        episodes_per_epoch=int(params['training']['val_tasks']),
                                        n_way=params['base']['n_way'],
                                        k_shot=params['base']['k_shot'],
                                        q_queries=params['base']['q_queries'],
                                        num_tasks=1,
                                        seed=seed),
                                        num_workers=params['training']['num_workers'],
                                        collate_fn=val_coll)

    evalTaskloader = FastDataLoader(
        evaluation, batch_sampler= val_task_sampler(dataset=evaluation,
                                        episodes_per_epoch=int(params['training']['test_tasks']),
                                        n_way=params['base']['n_way'],
                                        k_shot=params['base']['k_shot'],
                                        q_queries=params['base']['q_queries'],
                                        num_tasks=1,
                                        seed=seed),
                                        num_workers=params['training']['num_workers'],
                                        collate_fn=val_coll)

    #########################
    # MODEL & BATCHING
    #########################
    # Grabs relevant model params
    mod = model_params[model_name]
    model = grab_dual_model(name=model_name, mod=mod, out_dim=num_train_classes).to(device, dtype=torch.double)

    train_batch_fn = prep_batch_fns[0](device, params['training']['trans_batch'])

    # Sets up N-way K-shot batching function 
    val_batch_fn = prep_batch_fns[1](params['base']['n_way'],
                                    params['base']['k_shot'],
                                    params['base']['q_queries'],
                                    device,
                                    params['training']['trans_batch'])

    #########################
    # OPTIMISATION
    #########################
    # Optmiser and scheduler set up
    meta_opt = optim.Adam(model.parameters(), lr=params['hyper']['initial_lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=meta_opt,
                                        mode='min',
                                        patience=params['hyper']['patience'],
                                        factor=params['hyper']['factor'],
                                        min_lr=params['hyper']['min_lr'])

    #########################
    # LOSS FUNCTIONS
    #########################
    # Generates a weighted or normal loss function
    # Weighted uses inverse class frequency
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

    # Need annother loss function for the val setting as diff number classes 
    val_loss_fn = nn.CrossEntropyLoss().to(device)

    #########################
    # FITTING CALL
    #########################
    final_acc, final_loss, final_acc_std = fit_ss(
                model=model,
                optimiser=meta_opt,
                scheduler=scheduler,
                loss_fns=[loss_fn, val_loss_fn],
                dataloaders=[train_loader, flat_trainloader, valTaskloader, evalTaskloader],
                prep_batch_fns=[train_batch_fn, val_batch_fn],
                meta_fit_function=eval_step,
                params=params,
                meta_func_kwargs={'n_way':params['base']['n_way'],
                                'k_shot':params['base']['k_shot'],
                                'q_queries':params['base']['q_queries'],
                                'device': device}
    )

    return final_acc, final_loss, final_acc_std
    