"""
The file contains the main meta fit function for meta-baseline along with the
    validation sub-function.
The pre and post accuracy/loss split can be ignored for metric learners. It is
    included for gradint based learners which have explicit inner loop optimisations
"""
###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import numpy as np
import pandas as pd

from tqdm import trange
from datetime import datetime

from utils import *

###############################################################################
# PROTONET FIT FUNCTION 
###############################################################################

def mb_fit(learner, optimiser, scheduler, loss_fn, dataloaders, prep_batch_fns,
        fit_functions, params, meta_func_kwargs):
    """The main fit function for the meta-learner

    Args:
        learner (Torch nn object): The actual leanring model to be trained
        optimiser (Torch optim object): The optimiser to be used for training
        scheduler (Torch optim object): The learning rate scheduler to be used
        loss_fn (Torch nn object): Loss function for training
        dataloaders (List[Torch data objects]): A list of the dataloaders, expect train/val/test
        prep_batch_fns (Functions): Th prep batch functions to be used
        fit_functions (Functions): The actual meta batch fitting functions to be called
        params (Dictionary): The control paramater dictionary
        meta_func_kwargs (Dictionary): Some unpacked control params needed sooner

    Returns:
        final_pre (float): The final avg pre-inner-loop-update accurayc over the test set
        final_post (float): The final avg post-inner-loop-update accuracy over the test set
        final_loss (float): The final avg loss on the test set 
    """

    # Unpack dataloaders
    train_batch, val_batch = prep_batch_fns
    trainLoader, valLoader, evalLoader = dataloaders
    train_fit_function, val_fit_function = fit_functions

    # Chooses what type of validation step to take and sets according functions
    if params['data']['variable']:
        validation_step = validation_step_variable
    else:
        validation_step = validation_step_fixed

    # Initialises paths and logger functionality
    extra_path_data, log_path, train_path, val_path = path_start_up(params)
    logger = logger_start_up(log_path)

    logger.info(f'All Params: {params}')

    # Set up fresh  dataframes for train and val metrics
    column_names = ['Pre_acc', 'Post_acc', 'Back_loss']
    train_df = pd.DataFrame(columns=column_names)
    val_df = pd.DataFrame(columns=column_names)

    # Starts counters for episodes, break functions and validation plateau
    episode = 0
    break_counter = 0
    plateau_counter = 0
    best_val_post = 0
    # A state that can be changed and saved in paths to indicate crashed or not
    state = 'FIN'

    # Initalise main loop and break functionality
    break_key = False
    logger.info('########## STARTING TRAINING ##########')
    main_loop = trange(1, params['training']['epochs']+1, file=sys.stdout,
                            desc='Experiment Loop')
    # Set up value display for the main loop
    main_loop.set_postfix({'Pre_train': 0, 'Post_train': 0, 'Pre_val':0, 'Post_val':0})

    # Iterating over total epochs and then over batches in trainLoader
    for epoch in main_loop:
        for batch_index, batch in enumerate(trainLoader):
            episode += 1

            # Prep batch and move to GPU
            x, y = train_batch(batch, params['training']['train_batch_size'])

            train_loss, train_pre, train_post, train_post_std = train_fit_function(model=learner,
                                                        optimiser=optimiser,
                                                        loss_fn=loss_fn,
                                                        x=x,
                                                        y=y,
                                                        train=True,
                                                        **meta_func_kwargs)


            # Runs a validation at the first episode and every 'spacing' after
            if episode == 1 or episode % params['training']['eval_spacing'] == 0:
                # Runs a validation step, getting same kind of metrics as train
                val_loss, val_pre, val_post, val_post_std = validation_step(valLoader=valLoader, 
                                                                model=learner,
                                                                optimiser=optimiser, 
                                                                prep_batch=val_batch,
                                                                fit_function=val_fit_function,
                                                                loss_fn=loss_fn, 
                                                                params=params,
                                                                **meta_func_kwargs)

                scheduler.step(val_loss)

                # Metric collecting
                val_data = {'Pre_acc': val_pre, 'Post_acc':val_post,
                                                    'Back_loss':val_loss}
                val_df = update_csv(val_path, val_df, val_data)
                logger.info(f"Episode {episode}:: Validation:  { {key : round(val_data[key], 2) for key in val_data} }")

                # If new val is better, save model
                if np.greater(val_post, best_val_post):
                    # Stores best post validation to compare against
                    best_val_post = val_post
                    plateau_counter = 0
                    # Want to also save the model that did best
                    best_model_path = save_model(extra_path_data, learner)



            # New data for the metric tracking and logging
            train_data = {'Pre_acc': train_pre, 'Post_acc':train_post,
                                                'Back_loss':train_loss}


            # Update the train and val dataframes, includes saving
            train_df = update_csv(train_path, train_df, train_data)

            # Updaing the main loop display output
            update_dict = {'Pre_train': round(train_pre, 2), 'Post_train': round(train_post,2),
                            'Pre_val':round(val_pre, 2), 'Post_val':round(val_post, 2)}
            main_loop.set_postfix(update_dict)
            logger.info(f"Episode {episode}:  { {key: update_dict.get(key) for key in ['Pre_train', 'Post_train']} }")

    filename_updates([train_path, val_path], state)
    logger.info('########## FINISHED TRAINING ##########')

    final_figures(extra_path_data, params, train_df, val_df)

    # Loads the best validation modle to use on the test set
    best_learner = load_model(best_model_path, learner)
    # Can reuse the validation loop to carry out final testing
    final_loss, final_pre, final_post, final_post_std = validation_step(valLoader=evalLoader, 
                                                    model=best_learner,
                                                    optimiser=optimiser, 
                                                    prep_batch=val_batch,
                                                    fit_function=val_fit_function,
                                                    loss_fn=loss_fn, 
                                                    params=params,
                                                    **meta_func_kwargs)

    # Stores teh final results as well as logging and printing the,
    final_results = {'Pre_acc': final_pre, 'Post_acc':final_post,
                                        'Back_loss':final_loss}
    print(f'Final Results: {final_results}')
    logger.info('########## Results on Test Set ##########')
    logger.info(final_results)
    return final_post, final_loss, final_post_std


###############################################################################
# FIXED VALIDATION FUCNTION
###############################################################################
def validation_step_fixed(valLoader, model, optimiser, prep_batch, fit_function, loss_fn, params, **meta_func_kwargs):
    """Function deals with taking a validation step for a fixed length dataset

    Args:
        valLoader (torch data object): The validation data loader with val tasks
        model (torch nn module): The meta model we are training
        optimiser (torch optim object): Optimiser for updating the model
        prep_batch (function): Function for sorting teh incoming data into suitable
             batches for the model
        fit_function (function): The meta update function being used
        loss_fn (torch nn object): The loss function to use
        params (dict): Dictionary containing all of the experiment variables

    Returns:
        float, float, float: The average loss as well as the evarge pre and post 
            validation accuracies 
    """
    total_loss = 0
    total_pre = 0
    total_post = 0

    all_vals = []

    num_batches = len(valLoader)

    for batch_index, batch in enumerate(valLoader):
        # Prep batch and move to GPU
        x, y = prep_batch(batch, 1)

        back_loss, avg_pre, avg_post, all_post = fit_function(model=model,
                                                    optimiser=optimiser,
                                                    loss_fn=loss_fn,
                                                    x=x,
                                                    y=y,
                                                    train=False,
                                                    **meta_func_kwargs)

        total_loss += back_loss
        total_pre += avg_pre
        total_post += avg_post

        for val in all_post:
            all_vals.append(val)

    loss = total_loss/num_batches
    pre = total_pre/num_batches
    post = total_post/num_batches

    return loss, pre, np.mean(all_vals), np.std(all_vals)


###############################################################################
# VARIABLE VALIDATION FUCNTION
###############################################################################
def validation_step_variable(valLoader, model, optimiser, prep_batch, fit_function, loss_fn, params, **meta_func_kwargs):
    """Function deals with taking a validation step for a variable length dataset

    Args:
        valLoader (torch data object): The validation data loader with val tasks
        model (torch nn module): The meta model we are training
        optimiser (torch optim object): Optimiser for updating the model
        prep_batch (function): Function for sorting teh incoming data into suitable
             batches for the model
        fit_function (function): The meta update function being used
        loss_fn (torch nn object): The loss function to use
        params (dict): Dictionary containing all of the experiment variables

    Returns:
        float, float, float: The average loss as well as the evarge pre and post 
            validation accuracies 
    """
    total_loss = 0
    total_pre = 0
    total_post = 0

    all_vals = []

    num_batches = len(valLoader)

    for batch_index, batch in enumerate(valLoader):
        # Prep batch and move to GPU
        x_s, x_q, q_nums, y = prep_batch(batch, 1)

        # Grabs torch device to deal with compile function data movement
        device = torch.device('cuda:' + str(params['base']['cuda']) if \
            torch.cuda.is_available() else 'cpu')

        back_loss, avg_pre, avg_post, all_post = fit_function(model=model,
                                                    optimiser=optimiser,
                                                    loss_fn=loss_fn,
                                                    x_support=x_s,
                                                    x_query=x_q,
                                                    q_num=q_nums,
                                                    y=y,
                                                    device=device,
                                                    **meta_func_kwargs)

        total_loss += back_loss
        total_pre += avg_pre
        total_post += avg_post

        for val in all_post:
            all_vals.append(val)

    loss = total_loss/num_batches
    pre = total_pre/num_batches
    post = total_post/num_batches

    return loss, pre, np.mean(all_vals), np.std(all_vals)
