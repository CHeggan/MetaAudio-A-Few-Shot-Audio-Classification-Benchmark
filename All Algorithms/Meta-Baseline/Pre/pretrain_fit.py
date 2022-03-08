"""
The fitting function for pre-training of models. We have:
    -> train and validation splits being evaluated
    -> either a weighted or normal loss function
    -> Logging functionality
"""
################################################################################
# IMPORTS
################################################################################
import sys
import numpy as np
import pandas as pd

from Pre.pretrain_utils import *

from tqdm import trange
from datetime import datetime

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def catagorical_accuracy(targets, predictions):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def vote_catagorical_acc(targets, predictions):
    return (predictions == targets).sum().float() / targets.size(0)

###############################################################################
# MODEL PRETRAINING FIT FUNCTION
###############################################################################
def fit(model, optimiser, scheduler, loss_fn, dataloaders, prep_batch, params, device, variable):
    # Unpack dataloaders
    trainLoader, valLoader = dataloaders

    if variable:
        val_function = basic_val_var
    elif not variable:
        val_function = basic_val_fixed
    else:
        raise ValueError('What datatype is it then?')

    # Initialises paths and logger functionality
    extra_path_data, log_path, train_path, val_path = path_start_up(params)
    logger = logger_start_up(log_path)

    logger.info(f'All Params: {params}')

    # Set up fresh  dataframes for train and val metrics
    column_names = ['Acc', 'Back_loss']
    train_df = pd.DataFrame(columns=column_names)
    val_df = pd.DataFrame(columns=column_names)

    logger.info('########## STARTING TRAINING ##########')
    main_loop = trange(1, params['training']['epochs']+1, file=sys.stdout,
                            desc='Main Loop')
    # Set up value display for the main loop
    main_loop.set_postfix({'Train_Acc': 0, 'Val_Acc':0})

    best_val_accuracy = 0
    for epoch in main_loop:
            # Make sure model is in train mode
            model.train()
            train_accs = []
            train_loss = 0
            for batch_idx, batch in enumerate(trainLoader):
                x, y = prep_batch(batch)

                logits = model.forward(x, features=False)
                loss = loss_fn(logits, y)
                train_loss += loss.item()

                y_pred = logits.softmax(dim=1)
                acc = catagorical_accuracy(y, y_pred).cpu()
                train_accs.append(acc)

                loss.backward()
                optimiser.step()
            
            avg_train_acc = np.mean(train_accs)
            avg_train_loss = train_loss/(batch_idx + 1)
            
            # Performs a validation step, either normal or variable length
            avg_val_loss, avg_val_acc = val_function(model, valLoader, prep_batch, loss_fn, device)

            scheduler.step(avg_val_loss)


            # Metric collecting
            val_data = {'Acc': avg_val_acc, 'Back_loss':avg_val_loss}
            train_data = {'Acc': avg_train_acc, 'Back_loss':avg_train_loss}

            # Update the train and val dataframes, includes saving
            train_df = update_csv(train_path, train_df, train_data)
            val_df = update_csv(val_path, val_df, val_data)

            # Updaing the main loop display output
            update_dict = {'Train_Acc': round(avg_train_acc, 2), 'Val_Acc': round(avg_val_acc,2)}
            main_loop.set_postfix(update_dict)

            logger.info(f"Epoch {epoch}:  { {key: update_dict.get(key) for key in ['Pre_train', 'Post_train']} }")
            logger.info(f"Epoch {epoch}:: Validation:  { {key : round(val_data[key], 2) for key in val_data} }")

            # If new val post is cbetter, then save model
            if np.greater(avg_val_acc, best_val_accuracy):
                # Stores best post validation to compare against
                best_val_accuracy = avg_val_acc
                # Want to also save the model that did best
                best_model_path = save_model(extra_path_data, model)
    
    # Load best model and return to use in rest of meta baseline
    model = load_model(best_model_path, model)

    # Generate final figures
    final_figures_normal(extra_path_data, params, train_df, val_df)
    
    return best_model_path, model, best_val_accuracy


def basic_val_fixed(model, valLoader, prep_batch, loss_fn, device):
    model.eval()
    val_accs = []
    val_loss = 0
    for batch_idx, batch in enumerate(valLoader):
        x, y = prep_batch(batch)

        logits = model.forward(x, features=False)
        loss = loss_fn(logits, y)
        val_loss += loss.item()

        y_pred = logits.softmax(dim=1)
        val_acc = catagorical_accuracy(y, y_pred).cpu()
        val_accs.append(val_acc)
    
    avg_val_acc = np.mean(val_accs)
    avg_val_loss = val_loss/(batch_idx + 1)

    return avg_val_loss, avg_val_acc

def basic_val_var(model, valLoader, prep_batch, loss_fn, device):
    model.eval()
    val_accs = []
    losses = []

    for idx, batch in enumerate(valLoader):
        x_main, y_main = batch

        for i, sub_x in enumerate(x_main):
            mini_batch = (sub_x, y_main[i])
            x, y = prep_batch(mini_batch)

            logits = model.forward(x, features=False)
            scaled_y = torch.repeat_interleave(
                    y, x.shape[0]).to(device)

            loss = loss_fn(logits, scaled_y)
            losses.append(loss.item())

            soft_logits = logits.softmax(dim=1)
            y_preds = soft_logits.argmax(dim=1)

            final_pred, indices = torch.mode(y_preds)
            val_acc = (final_pred == y).sum().float()

            val_accs.append(val_acc.item())
        
        avg_val_acc = np.mean(val_accs)
        avg_val_loss = np.mean(losses)

    return avg_val_loss, avg_val_acc




    