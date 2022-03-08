"""
Meta-baseline works by utilising two training stages:
    -> base class conventional training with early stopping on some train/val
        split of D_base
    -> N-way K-shot fine-tuning with early stopping and validation selection
        based on teh actual validation split

Baselooper is different here as instead of the randomness being mainly placed
    on this sie, it is based on the pretraining of the base class classification
    model.

The model loaded in from the pre-training stage is the full model including
    logits however we dont use it at all and opt for the same bypass feature
    system used in SimpleShot, where a feature keyword is passed every forward
    call. 

We can load the seeds as they are used in the conventional training stage and use
    them for dataset splitting etc.
"""
###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import yaml
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange

from utils import set_seed, load_model, cuda_load_model
from MB_main import meta_baseline_main

from Pre.Main_pretrain import main as pretrain_main
from Pre.model_selection import *

###############################################################################
# MAIN RUN
###############################################################################
if __name__ =='__main__':
    # Loads in other expeirment params
    with open("mb_params.yaml") as stream:
        params = yaml.safe_load(stream)

    # Experiment variables
    NUM_REPEATS = params['base']['num_repeats']

    # Setting of cuda device
    device = torch.device('cuda:' + str(params['base']['cuda']) if \
        torch.cuda.is_available() else 'cpu')
    
    # Grabs the model name list for looping
    model_list = params['models']

    og_task_type = params['base']['task_type']

    # Iterate over model types
    for mod in model_list:
        print('\n\n\n\n\n')
        print(device)
        print(f'Starting Model: {mod}')

        # Generate new task type name and store it 
        task_type = og_task_type + '_' + mod + '_' + params['data']['norm'] + \
            '_' + str(NUM_REPEATS) + '_runs'
        params['base']['task_type'] = task_type

        # Perform multiple experiments 
        stored_results = []
        model_loop = trange(1, NUM_REPEATS+1, file=sys.stdout, desc=mod)
        for i in model_loop:
            # Generate a random seed with upper limit 1000 and sets it for modules
            seed = np.random.randint(low=0, high=1000)

            set_seed(seed)
            params['base']['seed'] = seed

            # best_model_path, model, best_val = pretrain_main(main_params=params, 
            #                                         model_name=mod)
            
            model_name = mod
            with open("Pre/models/params/all_model_params.yaml") as stream:
                model_params = yaml.safe_load(stream)
            print(model_params)
            print(f'Using: {model_name}')
            model = grab_model(model_name, model_params[model_name], out_dim=876)
            model = model.to(device, dtype=torch.double)

            model = cuda_load_model('Pre/best_val_model__03_02__11_26.pt', model, device)

            # We add a scaling factor to the model 
            temp = params['mb_specific']['scaling_factor']
            if params['mb_specific']['learnable']:
                model.temp = nn.Parameter(torch.tensor(temp)).to(device, dtype=torch.double)
            else:
                model.temp = temp

            # Move onto meta fine-tuning and meta evaluation
            post, loss, post_std = meta_baseline_main(params=params, 
                                                model=model, 
                                                device=device)

            stored_results.append(post)
            f = open(params['base']['task_type'] + ".txt", "a")
            f.write(f'{str(post)}, {str(post_std)}\n')
            f.close()

        stored_results = np.array(stored_results)
        f = open(params['base']['task_type'] + ".txt", "a")
        f.write(f'AVG: {str(np.mean(stored_results))}, STD: {str(np.std(stored_results))}\n')
        f.close()