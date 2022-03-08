"""
Script contains all required prep batch functions for fixed and variable batch
    functions. Includes:
        -> Normal fixed length batcher with trans keywork for transpose operation
        -> Variable length train and eval prep batch functions, both with trans
            keyword for transpose operation
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import numpy as np

###############################################################################
# FIXED LENGTH BATCHING FUNCTION (TRAIN AND EVAL)
###############################################################################
def prep_batch_fixed(n_way, k_shot, q_queries, device, trans):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        n_way (int)): The number of ways in classification task
        k_shot (int): Number of support vectors in a given task
        q_queries (int): Number of query vectors in a given task
        device (torch CUDA object): The CUDA device we want to load data to
        trans (boolean): Whether to apply transformer specific changes to data batching
    """
    def prep_batch_fixed(batch, meta_batch_size):
        """The child prep batch function. Takes some batch and processes it
            into proper tasks before moving it to a GPU for calculations.

        Args:
            batch (Tensor): The unformatted batch of data and tasks
            meta_batch_size (int): The expected batch size

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        x = x.reshape(meta_batch_size, (n_way*k_shot + n_way*q_queries),
                            x.shape[-2], x.shape[-1])

        # If transformer batch, we transpose and dont unsqueeze for channel 1
        if trans:
            x = torch.transpose(x, 2, 3)
            x = x.double().to(device)
        else:
            x = x.unsqueeze(2).double().to(device)

        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))

        # Creates a batch dimension and then repeats across it
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)

        y = y.long().to(device)

        return x, y
    return prep_batch_fixed

###############################################################################
# VARIABLE LENGTH TRAIN BATCH FUNCTION
###############################################################################
def prep_var_train(n_way, k_shot, q_queries, device, trans):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        n_way (int)): The number of ways in classification task
        k_shot (int): Number of support vectors in a given task
        q_queries (int): Number of query vectors in a given task
        device (torch CUDA object): The CUDA device we want to load data to
        trans (boolean): Whether to apply transformer specific changes to data batching
    """
    def prep_var_train(batch, meta_batch_size):
        """The child prep batch fucntion. Takes some batch and processes it
            into prper tasks before moving it to a GPU for calculations. Works
            specifically for teh variable length sets for training

        Args:
            batch (Tensor): The unformatted batch of data and tasks
            meta_batch_size (int): The expected batch size

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        # Creates a fresh zero tensor to fill with sampled x values
        new_x = torch.zeros(meta_batch_size*(n_way*k_shot + n_way*q_queries), x[0].shape[-2], x[0].shape[-1])
        # Iterates over our current list of multi dim x values
        for idx, samples in enumerate(x):
            # Samples one of teh available snippets and fills in teh new tensor
            ind = np.random.choice(samples.shape[0])
            new_x[idx] = samples[ind]

        # Reshapes teh new tensor into the expected dimensionality
        x = new_x.reshape(meta_batch_size, (n_way*k_shot + n_way*q_queries),
                            x[0].shape[-2], x[0].shape[-1] )

        
        # If transformer batch, we transpose and dont unsqueeze for channel 1
        if trans:
            x = torch.transpose(x, 2, 3)
            x = x.double().to(device)
        else:
            x = x.unsqueeze(2).double().to(device)

        # Generates and sets up the y value arary
        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))
        # Creates a batch dimension and then repeats across it
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)

        # Changes data type and moves to passed device(CUDA)
        y = y.long().to(device)

        return x, y
    return prep_var_train

###############################################################################
# VARIABLE LENGTH EVAL BATCH FUNCTION
###############################################################################
def prep_var_eval(n_way, k_shot, q_queries, device, trans):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        n_way (int)): The number of ways in classification task
        k_shot (int): Number of support vectors in a given task
        q_queries (int): Number of query vectors in a given task
        device (torch CUDA object): The CUDA device we want to load data to
        trans (boolean): Whether to apply transformer specific changes to data batching
    """
    def prep_var_eval(batch, meta_batch_size):
        """The child prep batch function. Takes some batch and processes it
            into proper tasks before moving it to a GPU for calculations. Works
            for the variable length sets at eval/test time

        Args:
            batch (Tensor): The unformatted batch of data and tasks
            meta_batch_size (int): The expected batch size

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        end_index = 0
        supports, queries = [], []
        for i in range(meta_batch_size):
            for idx, samples in enumerate( x[ end_index : end_index + (n_way*k_shot) ] ):
                supports.append(samples)
            end_index += n_way*k_shot
            for idx, samples in enumerate( x[ end_index : end_index + (n_way*q_queries) ] ):
                queries.append(samples)
            end_index += n_way*q_queries

        x_support = torch.zeros(meta_batch_size*(n_way*k_shot), x[0].shape[-2], x[0].shape[-1])
        for idx, samples in enumerate(supports):
            ind = np.random.choice(samples.shape[0])
            x_support[idx] = samples[ind]
        x_support = x_support.reshape(meta_batch_size, (n_way*k_shot), x[0].shape[-2], x[0].shape[-1])

        x_queries = torch.zeros(1, x[0].shape[-2], x[0].shape[-1])
        query_sample_nums = []
        for idx, samples in enumerate(queries):
            #print(samples.shape)
            query_sample_nums.append(samples.shape[0])
            for j, samp in enumerate(samples):
                if samp.ndim == 2:
                    samp = samp.unsqueeze(0)
                x_queries = torch.cat((x_queries, samp), 0)

        # Cuts the first sample as it was just zeros
        x_queries = x_queries[1:]

        # Generates and sets up the y value arary
        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))
        # Creates a batch dimension and then repeats across it
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)

        # Changes data type and moves to passed device(CUDA)
        y = y.long().to(device)

        if trans:
            x_support = torch.transpose(x_support, 2, 3)
            x_queries = torch.transpose(x_queries, 1, 2)
            x_support = x_support.double().to(device)
            x_queries = x_queries.double().to(device)
        else:
            x_support = x_support.unsqueeze(2).double().to(device)
            x_queries = x_queries.unsqueeze(1).double().to(device)

        return x_support, x_queries, query_sample_nums, y
    return prep_var_eval