"""

"""
###############################################################################
# IMPORTS
###############################################################################
import yaml
import torch
import random
import numpy as np
import learn2learn as l2l
from torch import nn, optim

from fit import fit
from meta_steps import meta_step_fixed, meta_step_var
from task_sampling_classes import NShotTaskSampler

from dataset_.SetupClass import DatasetSetup
from dataset_.DatasetClasses import NormDataset, FastDataLoader,TrainingVariableDataset

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
def main(params, model, device, batch_fns, seed):
    # Grabs the dataset splits if we have fixed class experiment
    class_splits = np.array(None)
    if params['data']['fixed']:
        class_splits = np.load(params['data']['fixed_path'], allow_pickle=True)
    # Grabs the split ratios for the classes 
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
        eval_step = meta_step_var
        train_dataset = TrainingVariableDataset
        val_dataset = NormDataset
        train_task_sampler = NShotTaskSampler
        val_task_sampler = NShotTaskSampler


    else:
        train_dataset = NormDataset
        val_dataset = NormDataset
        train_coll = None
        val_coll = None
        eval_step = meta_step_fixed
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


    # We start with a first order MAML for additional stabiltiy
    maml = l2l.algorithms.MAML(model, lr=params['hyper']['inner_lr'], first_order=True)

    # We start with a firts order MAML for additional stabiltiy
    meta_opt = optim.Adam(maml.parameters(), lr=params['hyper']['meta_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_opt,
                                        T_max=params['hyper']['T_max'],
                                        eta_min=params['hyper']['min_lr'])


    loss_fn = nn.CrossEntropyLoss().to(device)

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

    pre, post, loss, post_std = fit(
                    learner=maml,
                    optimiser=meta_opt,
                    scheduler=scheduler,
                    warm_up_episodes=params['training']['warm_up'],
                    loss_fn=loss_fn,
                    dataloaders=[backTaskloader, valTaskloader, evalTaskloader],
                    prep_batch_fns=[prep_train, prep_val],
                    fit_functions=[meta_step_fixed, eval_step],
                    params=params,
                    meta_func_kwargs={'n_way':params['base']['n_way'],
                                    'k_shot':params['base']['k_shot'],
                                    'q_queries':params['base']['q_queries']}
    )

    return pre, post, loss, post_std