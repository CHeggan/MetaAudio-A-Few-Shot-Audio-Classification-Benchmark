import os
import sys
import torch
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime


###############################################################################
# SEEDING
###############################################################################
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

###############################################################################
# UPDATE CSV FILE
###############################################################################
def update_csv(path, df, new_data):
    """
    Takes some df and some new data, adds in teh new data and overwrites the
        current instance of the file
    """
    # If no new data, we just rewrite file
    new_df = df.append(new_data, ignore_index=True)
    new_df.to_csv(path)
    return new_df

###############################################################################
# LOGGER INNITIALISATION
###############################################################################
def logger_start_up(file_path):
    """
    Creates the logger object for any particualr run of the code, also creates
        the file that all this data will be stored in
    """
    logging.basicConfig(filename=file_path, format='%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%d/%m/%y %I.%M.%S %p', level=logging.INFO,
                            filemode='w')

    logger = logging.getLogger(__name__)
    return logger

###############################################################################
# PATH INITIALISATION
###############################################################################
def path_start_up(params):
    """
    Initialises the log, train and val paths for all the data form the run to be
        stored
    """
    # Adds on identifies for task type, total episodes and seed
    add_on = (params['base']['task_type'] + '_' + str(params['training']['epochs']) + '_seed_' +
                str(params['base']['seed']))
    # Grabs time and puts into wanted format
    now = datetime.now()
    now_path = now.strftime("_%d_%m__%H_%M")

    # All relveant identifiers except time are in the parent directory name
    result_dir = os.path.join('Pre/results', params['base']['task_type'], add_on)

    # Task dircetory creation first
    task_dir = os.path.join('Pre/results', params['base']['task_type'])

    if os.path.exists(task_dir):
        pass
    else:
        try:
            os.mkdir(task_dir)
        except OSError:
            print ("Creation of the directory %s failed" % task_dir)
            sys.exit()
        else:
            print ("Successfully created the directory %s" % task_dir)

    # Now try to create the more specific results directory
    if os.path.exists(result_dir):
        pass
    else:
        try:
            os.mkdir(result_dir)
        except OSError:
            print ("Creation of the directory %s failed" % result_dir)
            sys.exit()
        else:
            print ("Successfully created the directory %s" % result_dir)

    # Paths to the trianing and validation folders
    train_path = os.path.join(result_dir, 'train_metrics_' + now_path + '.csv')
    val_path = os.path.join(result_dir, 'val_metrics_' + now_path + '.csv')

    # File name for logs, includes task type and most other details
    log_file = 'PRE' + '_' + add_on + now_path + '.log'
    # Actual path to teh log file
    log_path = os.path.join('Pre/logs', log_file)

    return [result_dir, now_path], log_path, train_path, val_path

###############################################################################
#F INAL PATH UPDATES
###############################################################################
def filename_updates(paths, state):
    """
    Updates the train and val csv data paths to indicate whether it finished
        or crashed
    """
    for path in paths:
        new = path.split('.')[0] + '_' + state + '.' + path.split('.')[1]
        os.rename(path, new)


###############################################################################
# FINAL FIGURE CREATION (PRETRAINIG)
###############################################################################
def final_figures_normal(extra_path_data, params, train_df, val_df):
    results_dir, now_path = extra_path_data

    norm_path = os.path.join(results_dir, params['base']['task_type'] + now_path + '.svg')
    df_path = os.path.join(results_dir, params['base']['task_type'] + 'graph_data_' + now_path + '.csv')

    train_acc = train_df['Acc'].values
    val_acc = val_df['Acc'].values

    # Normal plots, of the actual data avilable
    plt.plot(train_acc)
    plt.plot(val_acc)

    plt.ylim(0, 1)
    plt.xlim(0, len(train_df['Acc']))
    plt.legend(['Train Acc', 'Val Acc'])
    plt.savefig(norm_path)
    plt.clf()

###############################################################################
# SAVE MODEL
###############################################################################
def save_model(extra_path_data, model):
    results_dir, now_path = extra_path_data
    path = os.path.join(results_dir, 'best_val_model_' + now_path + '.pt')
    torch.save(model.state_dict(), path)
    return path

###############################################################################
# LOAD MODEL (NORMAL)
###############################################################################
def load_model(model_path, base_model):
    base_model.load_state_dict(torch.load(model_path))
    return base_model
