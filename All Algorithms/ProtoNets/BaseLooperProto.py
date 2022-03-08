"""
Scripts loops over base learner models in a sequential fashion. Aim is to have the
    code be started and left for multiple days while it generates results for 
    many different experiments

Each instantiation:
    -> Loops over relevant models 
    -> generates unique task IDs based on experiment params
    -> Only considers a single dataset at a time
    -> Runs n repeat experiments of each learner w/ random seeding, each with 
        at least > 5000 final test tasks(aim is 10k). The reason for so many end task here
        is for proper evaluation
    -> Overrides some variables in the params dictionary loaded from proto_params.yaml

This version of the code is set to be used for either variable or fixed length
    sets, this change means that there are additional components:
        -> Variable length prep batch functions for training and evaluation
        -> Variable raw to spectrogram dataset class that does both training
            and evaluation
"""
###############################################################################
# IMPORTS
###############################################################################
import sys
import yaml
import torch
import random
import numpy as np

from tqdm import trange
from utils_proto import set_seed
from ProtoMain import single_run_main
from model_selection import grab_model
from all_prep_batches import prep_batch_fixed, prep_var_eval, prep_var_train


###############################################################################
# MAIN RUUN
###############################################################################
if __name__ == '__main__':
    #########################
    # PRELIMINARIES
    #########################
    # Loads in other expeirment params
    with open("proto_params.yaml") as stream:
        params = yaml.safe_load(stream)

    # Loads in model params
    with open("models/params/all_model_params.yaml") as stream:
        model_params = yaml.safe_load(stream)

    print(model_params)

    # Setting of cuda device
    device = torch.device('cuda:' + str(params['base']['cuda']) if \
        torch.cuda.is_available() else 'cpu')

    # Get number of repeat experiments to run
    NUM_REPEATS = params['base']['num_repeats']
    og_task = params['base']['task_type']

    # Number of dimensions for models to output to
    out_dim = params['base']['out_dim']
    
    #########################
    # BATCH FUNCTIONS
    #########################
    if params['data']['variable']:
        batch_funcs = [prep_batch_fixed, prep_var_eval]
    elif params['data']['variable'] == False:
        batch_funcs = [prep_batch_fixed, prep_batch_fixed]
    else:
        raise ValueError('What kind of data is it then?')

    # Grabs the model name list for looping
    model_list = params['models']

    # Iterate over model types
    for mod in model_list:
        print('\n\n\n\n\n')
        print(device)
        print(f'Starting Model: {mod}')

        # Generate new task type name and store it 
        task_type = og_task + mod + '_' + params['data']['norm'] + '_' + str(NUM_REPEATS) + \
            '_runs'
        params['base']['task_type'] = task_type

        # Perform multiple experiments 
        stored_results = []
        model_loop = trange(1, NUM_REPEATS+1, file=sys.stdout, desc=mod)
        for i in model_loop:
            # Generate a random seed with upper limit 1000 and sets it for modules
            seed = np.random.randint(low=0, high=1000)
            set_seed(seed)
            params['base']['seed'] = seed

            # Grabs relevant model params and the torch model itself
            mod_params = model_params[mod]
            print(mod_params)
            model = grab_model(mod, mod_params, out_dim)
            model = model.to(device, dtype=torch.double)

            pre, post, loss, post_std = single_run_main(params=params, 
                                                model=model, 
                                                device=device,
                                                batch_fns=batch_funcs,
                                                seed=seed)

            
            stored_results.append(post)
            f = open(params['base']['task_type'] + ".txt", "a")
            f.write(f'{str(post)}, {str(post_std)}\n')
            f.close()

        stored_results = np.array(stored_results)
        f = open(params['base']['task_type'] + ".txt", "a")
        f.write(f'AVG: {str(np.mean(stored_results))}, STD: {str(np.std(stored_results))}\n')
        f.close()

