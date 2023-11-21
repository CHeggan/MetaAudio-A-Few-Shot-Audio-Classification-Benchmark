"""
Script hosts the fitting functions for the SimpleShot algorithm. This includes:
    -> Main fitting function
    -> Validation step for val and 
"""

##############################################################################
# IMPORTS
##############################################################################

import sys
import time
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import datetime
from prep_batch_functions import prep_batch_conventional

from utils import *
from metric import catagorical_accuracy

##############################################################################
# SIMPLESHOT FIT FUNCTION
##############################################################################
def fit_ss(model, optimiser, scheduler, loss_fns, dataloaders, prep_batch_fns,
    meta_fit_function, params, meta_func_kwargs):

    # Unpacks loss functions
    train_loss_fn, val_loss_fn = loss_fns
    # Unpacks prep batch functions
    train_batch_fn, val_batch_fn = prep_batch_fns
    # Unpacks data loaders
    trainloader, flat_trainloader, valloader, testloader = dataloaders

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
    column_names = ['Acc', 'Loss']
    train_df = pd.DataFrame(columns=column_names)
    val_df = pd.DataFrame(columns=column_names)

    # Starts metric for best validation score
    best_val_post = 0

    # normal batched training fitting
    logger.info('########## STARTING TRAINING ##########')
    main_loop = trange(1, params['training']['epochs']+1, file=sys.stdout,
                            desc='Base Loop')
    # Set up value display for the main loop
    main_loop.set_postfix({'Train Acc': 0, 'Train Loss':0, 'Val Acc': 0, 'Val Loss':0})


    avg_val_acc = 0
    avg_val_loss = 0
    # Iterating over total epochs and then over batches in trainLoader

    for epoch in main_loop:
        model.train()
        train_losses = []
        train_accs = []

        for _ in range(params['training']['eval_batch_spacing']):
            batch = next(iter(trainloader))
            x, y = train_batch_fn(batch)

            logits = model.forward(x, features=False)
            loss = train_loss_fn(logits, y)
            train_losses.append(loss.item())

            y_pred = logits.softmax(dim=1)
            acc = catagorical_accuracy(y, y_pred).cpu()
            train_accs.append(acc)

            loss.backward()
            optimiser.step()


        avg_train_acc = np.mean(train_accs)
        avg_train_loss = np.mean(train_losses)

        
        # Cycle through our flattened train data to get avg feature representations
        out_features = []
        with torch.no_grad():
            model.eval()
            for _, batch in enumerate(flat_trainloader):

                # Need to do some extra unpacking if data is variable due to collation function =
                if params['data']['variable']:
                    x_main, y_main = batch

                    for i, sub_x in enumerate(x_main):
                        mini_batch = (x_main[i], y_main)
                        x_sub, y_sub = train_batch_fn(mini_batch)
                        
                        logits, data_features = model.forward(x_sub, features=True)
                        batch_avg = torch.mean(data_features, dim=0).to('cpu').detach().numpy()
                        out_features.append(batch_avg)

                else:
                    x, y = train_batch_fn(batch)
                    logits, data_features = model.forward(x, features=True)
                    batch_avg = torch.mean(data_features, dim=0).to('cpu').detach().numpy()
                    out_features.append(batch_avg)


        # Convert feature calculations to tensor and reduce
        out_features = torch.Tensor(np.array(out_features))
        avg_features = torch.mean(out_features, dim=0)

        # Run a validation step 
        avg_val_loss, avg_val_acc, val_std = validation_step(valLoader=valloader,
                                                    model=model,
                                                    avg_features=avg_features,
                                                    prep_batch=val_batch_fn,
                                                    fit_function=meta_fit_function,
                                                    loss_fn=val_loss_fn,
                                                    params=params,
                                                    func_kwargs=meta_func_kwargs)

        # Perform scheduler step after validation
        scheduler.step(avg_val_loss)

        # If new val is better, save model
        if np.greater(avg_val_acc, best_val_post):
            # Stores best post validation to compare against
            best_val_post = avg_val_acc
            # Want to also save the model that did best
            best_model_path = save_model(extra_path_data, model)
            best_features_path = save_features(extra_path_data, avg_features)

        # Metric collecting
        val_data = {'Acc':avg_val_acc, 'Loss':avg_val_loss}
        # New data for the metric tracking and logging
        train_data = {'Acc':avg_train_acc,'Loss':avg_train_loss}

        # Update the train and val dataframes, includes saving
        val_df = update_csv(val_path, val_df, val_data)
        train_df = update_csv(train_path, train_df, train_data)

        # Updaing the main loop display output
        update_dict = {'Train Acc': round(avg_train_acc, 2), 'Train Loss': round(avg_train_loss, 2), 
                        'Val Acc': round(avg_val_acc, 2), 'Val Loss':round(avg_val_loss, 2)}
        main_loop.set_postfix(update_dict)

        # Update thw log file
        logger.info(f"Epoch {epoch}:  { {key: update_dict.get(key) for key in ['Pre_train', 'Post_train']} }")
        logger.info(f"Epoch {epoch}:: Validation:  { {key : round(val_data[key], 2) for key in val_data} }")

    logger.info('########## FINISHED TRAINING ##########')
    final_figures_normal(extra_path_data, params, train_df, val_df)
    
    # Loads the best validation modle to use on the test set
    best_model_path = 'best_val_model__29_11__19_20.pt'
    best_features_path = 'best_features__29_11__19_20.npy'
    best_learner = load_model(best_model_path, model)
    best_features = np.load(best_features_path, allow_pickle=True)

    # Can reuse the validation loop to carry out final testing
    avg_test_loss, avg_test_acc, test_std = validation_step(valLoader=testloader,
                                            model=best_learner,
                                            avg_features=best_features,
                                            prep_batch=val_batch_fn,
                                            fit_function=meta_fit_function,
                                            loss_fn=val_loss_fn,
                                            params=params,
                                            func_kwargs=meta_func_kwargs)

    # Stores teh final results as well as logging and printing the,
    final_results = {'Test Acc': avg_test_acc, 'Test Loss':avg_test_loss}

    print(f'Final Results: {final_results}')
    # logger.info('########## Results on Test Set ##########')
    # logger.info(final_results)
    return avg_test_acc, avg_test_loss, test_std
    

###############################################################################
# FIXED VALIDATION FUCNTION
###############################################################################
def validation_step_fixed(valLoader, model, avg_features, prep_batch, fit_function, loss_fn, params, func_kwargs):
    """Function deals with taking a validation step for a fixed length dataset

    Args:
        valLoader (torch data object): The validation data loader with val tasks
        model (torch nn module): The meta model we are training
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

        back_loss, avg_post, all_post = fit_function(model=model,
                                                    params=params,
                                                    loss_fn=loss_fn,
                                                    avg_features=avg_features,
                                                    x=x,
                                                    y=y,
                                                    **func_kwargs)

        total_loss += back_loss
        total_post += avg_post

        for val in all_post:
            all_vals.append(val)

    loss = total_loss/num_batches
    post = total_post/num_batches

    return loss, np.mean(all_vals), np.std(all_vals)

###############################################################################
# VARIABLE VALIDATION FUCNTION
###############################################################################
def validation_step_variable(valLoader, model, avg_features, prep_batch, fit_function, loss_fn, params, func_kwargs):
    """Function deals with taking a validation step for a variable length dataset

    Args:
        valLoader (torch data object): The validation data loader with val tasks
        model (torch nn module): The meta model we are training
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
    total_post = 0

    all_vals = []

    num_batches = len(valLoader)

    for batch_index, batch in enumerate(valLoader):
        # Prep batch and move to GPU
        x_s, x_q, q_nums, y = prep_batch(batch, 1)

        back_loss, avg_post, all_post = fit_function(model=model,
                                                params=params,
                                                loss_fn=loss_fn,
                                                avg_features=avg_features,
                                                x_support=x_s,
                                                x_query=x_q,
                                                q_num=q_nums,
                                                y=y,
                                                **func_kwargs)

        total_loss += back_loss
        total_post += avg_post

        for val in all_post:
            all_vals.append(val)

    loss = total_loss/num_batches
    post = total_post/num_batches

    return loss, np.mean(all_vals), np.std(all_vals)



###############################################################################
# SAVE BEST FEATURES
###############################################################################
def save_features(extra_path_data, features):
    results_dir, now_path = extra_path_data
    path = os.path.join(results_dir, 'best_features_' + now_path + '.npy')
    np.save(path, features, allow_pickle=True)
    return path

