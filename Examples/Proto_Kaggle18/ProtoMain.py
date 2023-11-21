""" 
The main call file, does the following:
    -> Unpacks control parameters and sends much of them where they are needed
    -> Runs the data setup class which work out norm stats etc
    -> Creates the N-shot data loaders for train/val/test
    -> Initialises the prep batch nested function
    -> Calls the main fit function
"""


###############################################################################
# IMPORTS
###############################################################################
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim, distributions as dist

from fit_proto import fit
from proto_steps import proto_step_fixed, proto_step_var
from task_sampling_classes import NShotTaskSampler

from dataset_.SetupClass import DatasetSetup
from dataset_.DatasetClasses import NormDataset, FastDataLoader, TrainingVariableDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
def single_run_main(params, model, device, batch_fns, seed):
    # Grabs the class splits if we want to used the fixed split option
    class_splits = None
    if params['data']['fixed']:
        class_splits = np.load(params['data']['fixed_path'], allow_pickle=True)

    # Grabs the dataset splits
    splits = [params['split']['train'], params['split']['val'], params['split']['test']]

    # Runs the setup class for the dataset, this generates splits, norm stats and proper pathing
    # Runs the setup class for the dataset
    setup = DatasetSetup(params=params,
                        splits=splits,
                        seed=seed,
                        class_splits=class_splits)

# Defines the datasets to be used
    if params['data']['variable']:
        train_coll = None
        val_coll = my_collate
        eval_step = proto_step_var

        train_dataset = TrainingVariableDataset
        val_dataset = NormDataset
        train_task_sampler = NShotTaskSampler
        val_task_sampler = NShotTaskSampler


    else:
        train_dataset = NormDataset
        val_dataset = NormDataset
        train_coll = None
        val_coll = None
        eval_step = proto_step_fixed
        train_task_sampler = NShotTaskSampler
        val_task_sampler = NShotTaskSampler



    background = train_dataset(data_path=params['data']['data_path'],
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

    val_batch_size = 1 if params['data']['variable'], else params['training']['train_batch_size']
    # Actually creates the dataloaders
    backTaskloader = FastDataLoader(
        background, batch_sampler= train_task_sampler(dataset=background,
                                        episodes_per_epoch=params['training']['episodes_per_epoch'],
                                        n_way=params['base']['n_way'],
                                        k_shot=params['base']['k_shot'],
                                        q_queries=params['base']['q_queries'],
                                        num_tasks=params['training']['train_batch_size'],
                                        seed=seed),
                                        num_workers=params['training']['num_workers'],
                                        collate_fn=train_coll)

    valTaskloader = FastDataLoader(
        validation, batch_sampler= val_task_sampler(dataset=validation,
                                        episodes_per_epoch=int(params['training']['val_tasks']),
                                        n_way=params['base']['n_way'],
                                        k_shot=params['base']['k_shot'],
                                        q_queries=params['base']['q_queries'],
                                        num_tasks=val_batch_size,
                                        seed=seed),
                                        num_workers=params['training']['num_workers'],
                                        collate_fn=val_coll)

    evalTaskloader = FastDataLoader(
        evaluation, batch_sampler= val_task_sampler(dataset=evaluation,
                                        episodes_per_epoch=int(params['training']['test_tasks']),
                                        n_way=params['base']['n_way'],
                                        k_shot=params['base']['k_shot'],
                                        q_queries=params['base']['q_queries'],
                                        num_tasks=val_batch_size,
                                        seed=seed),
                                        num_workers=params['training']['num_workers'],
                                        collate_fn=val_coll)

    # We start with a firts order MAML for additional stabiltiy
    meta_opt = optim.Adam(model.parameters(), lr=params['hyper']['initial_lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=meta_opt,
                                        mode='min',
                                        patience=params['hyper']['patience'],
                                        factor=params['hyper']['factor'],
                                        min_lr=params['hyper']['min_lr'])

    # Creates the loss function
    loss_fn = nn.NLLLoss().to(device)
    #loss_fn = nn.CrossEntropyLoss().to(device)

    # Sets up the prep batch functions
    prep_train = batch_fns[0](n_way=params['base']['n_way'],
                                k_shot=params['base']['k_shot'],
                                q_queries=params['base']['q_queries'],
                                device=device,
                                trans=params['training']['trans_batch'])
    prep_val = batch_fns[1](n_way=params['base']['n_way'],
                                k_shot=params['base']['k_shot'],
                                q_queries=params['base']['q_queries'],
                                device=device,
                                trans=params['training']['trans_batch'])

    # Runs the actual fitting function, contained in fit.py
    pre, post, loss, post_std = fit(
                    learner=model,
                    optimiser=meta_opt,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    dataloaders=[backTaskloader, valTaskloader, evalTaskloader],
                    prep_batch_fns=[prep_train, prep_val],
                    fit_functions=[proto_step_fixed, eval_step],
                    params=params,
                    meta_func_kwargs={'n_way':params['base']['n_way'],
                                    'k_shot':params['base']['k_shot'],
                                    'q_queries':params['base']['q_queries'],
                                    'distance':params['base']['distance']}

    )

    return pre, post, loss, post_std
