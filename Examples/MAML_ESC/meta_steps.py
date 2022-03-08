"""
Script contains teh main meta step functions needed for both fixed and variable 
    length datasets. In theoy it is possible to simply generalise these functions 
    however for a little addiitonal clarity they are kept distinct
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import numpy as np
import torch.nn as nn

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def catagorical_accuracy(targets, predictions):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def vote_catagorical_acc(targets, predictions):
    return (predictions == targets).sum().float() / targets.size(0)

def majority_vote(soft_logits, query_nums):
    y_preds = soft_logits.argmax(dim=1)

    end_index = 0
    aggregrated_preds = torch.zeros(len(query_nums))
    for idx, num in enumerate(query_nums):
        slice = y_preds[end_index:(end_index + num)]
        value, indices = torch.mode(slice)
        aggregrated_preds[idx] = value
        end_index += slice.shape[0]
    return aggregrated_preds


###############################################################################
# META STEP MAIN (FIXED)
###############################################################################
def meta_step_fixed(maml_model, optimiser, loss_fn, x, y, train, adapt_steps, 
        n_way, k_shot, q_queries):
    """Takes a gradient based meta-learning step using the l2l framework. Works
        specifically with fixed length samples

    Args:
        maml_model (torch object): The GBML wrapped base learner
        optimiser (torch object): Loss optimiser
        loss_fn (torch object): Loss functon to use
        x (tensor): Our unpacked x data from batch
        y (tensor): Unpacked y data from batch
        train (boolean): Should we train or eval?
        adapt_steps (int): Number of inner loop steps to take
        n_way (int): N-way classification number 
        k_shot (int): Number of support samples per N novel class
        q_queries (int): Number of query samples per N novel class 

    Returns:
        float, float, float: Final avg back loss, pre acc and post acc
    """
    optimiser.zero_grad()
    # Sets up metric tracking
    total_query_loss = 0
    pre_acc_total = 0
    post_acc_total = 0

    all_post = []

    for idx in range(x.shape[0]):
        cloned = maml_model.clone()

        meta_batch = x[idx]
        # Grabs the correct x data instances
        x_task_train = (meta_batch[:(n_way * k_shot)])
        x_task_val = (meta_batch[(n_way * q_queries):])

        y_task_train = (y[idx][:(n_way * k_shot)])
        y_task_val = (y[idx][(n_way * q_queries):])

        # Obtains the pre_trian accurayc on query samples
        with torch.no_grad():
            pre_query_logits = cloned(x_task_val)
            pre_query_loss = loss_fn(pre_query_logits, y_task_val)
            pre_query_pred = pre_query_logits.softmax(dim=1)
            pre_acc = catagorical_accuracy(y_task_val, pre_query_pred)

        # Adapts using the support vectors
        for _ in range(adapt_steps):
            mid_support_logits = cloned(x_task_train)
            mid_support_loss = loss_fn(mid_support_logits, y_task_train)
            cloned.adapt(mid_support_loss)

        if not train:
            with torch.no_grad():
                post_query_logits = cloned(x_task_val)
                post_query_loss = loss_fn(post_query_logits, y_task_val)
                post_query_pred = post_query_logits.softmax(dim=1)
                post_acc = catagorical_accuracy(y_task_val, post_query_pred)
        else:
                post_query_logits = cloned(x_task_val)
                post_query_loss = loss_fn(post_query_logits, y_task_val)

                post_query_pred = post_query_logits.softmax(dim=1)
                post_acc = catagorical_accuracy(y_task_val, post_query_pred)

        total_query_loss += post_query_loss
        pre_acc_total += pre_acc.item()
        post_acc_total += post_acc.item()

        all_post.append(post_acc.item())


    back_loss = total_query_loss /x.shape[0]
    avg_pre = pre_acc_total/x.shape[0]
    avg_post = post_acc_total/x.shape[0]

    if train:
        back_loss.backward()
        optimiser.step()

    # Makes sure model is in train at end of any given step
    maml_model.train()

    return back_loss.item(), avg_pre, np.mean(all_post), all_post

###############################################################################
# META STEP MAIN (VARIABLE)
###############################################################################
def meta_step_var(maml_model, optimiser, loss_fn, x_support, x_query, q_num, y, 
        adapt_steps, device, n_way, k_shot, q_queries):
    """Takes a gradient based meta-learning step using the l2l framework. Works
        specifically for variable length data

    Args:
        maml_model (torch object): The GBML wrapped base learner
        optimiser (torch object): Loss optimiser
        loss_fn (torch object): Loss functon to use
        x (tensor): Our unpacked x data from batch
        y (tensor): Unpacked y data from batch
        train (boolean): Should we train or eval?
        adapt_steps (int): Number of inner loop steps to take
        device (cuda object): CUDA device that we carry out experiment on
        n_way (int): N-way classification number 
        k_shot (int): Number of support samples per N novel class
        q_queries (int): Number of query samples per N novel class 

    Returns:
        float, float, float: Final avg back loss, pre acc and post acc
    """
    optimiser.zero_grad()
    # Sets up metric tracking
    total_query_loss = 0
    pre_acc_total = 0
    post_acc_total = 0

    all_post = []

    # Kepe track of the last used q_num access idx
    last_query_idx = 0
    # Iterate over num tasks
    for idx in range(x_support.shape[0]):
        cloned = maml_model.clone()

        x_task_train = x_support[idx]
        sub_q_num = q_num[idx*(n_way*q_queries): (idx+1)*(n_way*q_queries)]
        q_num_sub_sum = sum(sub_q_num)
        x_task_val = x_query[last_query_idx : (last_query_idx + q_num_sub_sum)]
        # Update tracking index
        last_query_idx += q_num_sub_sum 

        # y value access is same as a fixed length batching
        y_task_train = y[idx][:(n_way * k_shot)]
        y_task_val = y[idx][(n_way * q_queries):]

        # Obtains the pre_trian accurayc on query samples
        with torch.no_grad():
            pre_query_logits = cloned(x_task_val)
            # Take softmax
            pre_query_pred = pre_query_logits.softmax(dim=1)
            # Chnage sub nums ot tensor 
            sub_q_nums_tens = torch.tensor(sub_q_num).to(device)
            # Scale up clip labels for loss
            post_query_y = torch.repeat_interleave(y_task_val, sub_q_nums_tens).to(device)
            # Gets loss based on scaled uplogits
            pre_query_loss = loss_fn(pre_query_logits, post_query_y)

            pre_soft_logits = pre_query_logits.softmax(dim=1)
            pre_query_pred = majority_vote(pre_soft_logits, sub_q_nums_tens).to(device, dtype=torch.long)
            pre_acc = vote_catagorical_acc(y_task_val, pre_query_pred)

        # Adapts using the support vectors
        for _ in range(adapt_steps):
            mid_support_logits = cloned(x_task_train)
            mid_support_loss = loss_fn(mid_support_logits, y_task_train)
            cloned.adapt(mid_support_loss)

        # We scale up the y task val to directly compare for loss but use majority for acc
        with torch.no_grad():
            post_query_logits = cloned(x_task_val)
            sub_q_nums_tens = torch.tensor(sub_q_num).to(device)
            post_query_y = torch.repeat_interleave(y_task_val, sub_q_nums_tens).to(device)
            post_query_loss = loss_fn(post_query_logits, post_query_y)

            soft_logits = post_query_logits.softmax(dim=1)
            post_query_pred = majority_vote(soft_logits, sub_q_nums_tens).to(device, dtype=torch.long)
            post_acc = vote_catagorical_acc(y_task_val, post_query_pred)  

        total_query_loss += post_query_loss
        pre_acc_total += pre_acc.item()
        post_acc_total += post_acc.item()

        all_post.append(post_acc.item())

    back_loss = total_query_loss /x_support.shape[0]
    avg_pre = pre_acc_total/x_support.shape[0]
    avg_post = post_acc_total/x_support.shape[0]

    # Makes sure model is in train at end of any given step
    maml_model.train()

    return back_loss.item(), avg_pre, np.mean(all_post), all_post
